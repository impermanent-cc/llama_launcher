"""The Build tab: a CMake build-option catalog form (native or container
target), a live command/Containerfile preview, and a "Generate" action that
renders the build via core/build_command.py, writes a Containerfile for the
container target, and records a BuildOutput in the store registry. This tab
never runs podman/cmake itself -- see core/build_command.py and
store/builds.py for the renderers and store that do the actual work.
"""
import datetime
import shutil
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QRunnable, QThreadPool, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QGroupBox,
    QScrollArea, QPlainTextEdit, QPushButton, QApplication, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QMessageBox,
)

from llama_launcher.core.build_catalog import BUILD_CATALOG
from llama_launcher.core.build_command import (
    config_slug, auto_tag, render_native, render_container, default_images,
)
from llama_launcher.core.build_outputs import (
    OutputRow, classify_outputs, untracked_custom_tags, profiles_using,
)
from llama_launcher.core.build_spec import BuildConfig, BuildOutput
from llama_launcher.core.settings_catalog import for_engine
from llama_launcher.services.gpu import query_compute_caps
from llama_launcher.services.runtime import list_images_detailed, remove_image
from llama_launcher.store.builds import (
    builds_dir, save_build_config, list_build_configs, load_outputs,
    add_output, remove_output, write_containerfile, new_output_id,
)
from llama_launcher.store.profiles import list_profiles, save_profile
from llama_launcher.ui.widgets.no_wheel import NoWheelComboBox
from llama_launcher.ui.widgets.setting_widgets import make_widget, make_row_label


# BuildPanel is constructed with only a base_dir (no Profile/runtime context
# to read a container binary from, unlike ConfigurePanel/MonitorController),
# so podman calls here use a fixed default -- same default Runtime.binary
# uses everywhere else in the app (core/spec.py).
_BUILD_BINARY = "podman"


class _OutputsGather(QRunnable):
    """Off-UI-thread body of refresh_outputs(): the only slow part is the
    `podman images` subprocess call, so that's all this does. Same delivery
    model as ConfigurePanel's _CheckFitGather/_FitGpusGather -- write the
    result onto a plain attribute of the owning BuildPanel (atomic under the
    GIL) instead of a cross-thread Qt signal, and hold a reference to the
    panel so it can't be garbage-collected mid-gather. The actual
    classify+render work happens back on the UI thread in refresh_outputs_sync,
    called by _poll_outputs once _outputs_inflight flips back to False.
    """
    def __init__(self, owner, binary: str):
        super().__init__()
        self._owner = owner
        self._binary = binary

    def run(self):
        try:
            images = list_images_detailed(self._binary)
        except Exception:            # noqa: BLE001 - worker must never raise
            images = {}
        self._owner._outputs_images_result = images
        self._owner._outputs_inflight = False


class BuildPanel(QWidget):
    """Owns all Build-tab widgets. `base_dir` is the store directory (a
    profiles/builds root), passed explicitly rather than resolved internally
    so tests never touch the user's real config dir.
    """

    def __init__(self, *, base_dir, parent=None):
        super().__init__(parent)
        self.base_dir = Path(base_dir)
        self._widgets: dict[str, object] = {}
        self._group_boxes: dict[str, QGroupBox] = {}
        self._saved_configs: dict[str, BuildConfig] = {}
        self._outputs_rows: list[OutputRow] = []
        self._outputs_inflight = False
        self._outputs_images_result: dict = {}

        root = QVBoxLayout(self)

        body = QHBoxLayout()
        root.addLayout(body, 1)

        # LEFT: config identity + engine/target + source/image fields
        left = QGroupBox("Build config")
        self._left_form = QFormLayout(left)
        self.name_edit = QLineEdit()
        self.name_edit.setToolTip("Name for this build config; also the slug "
                                   "used for the saved-config file and any "
                                   "Containerfile it writes.")
        self.name_edit.textChanged.connect(self.refresh_preview)
        self._left_form.addRow("Name", self.name_edit)

        self.config_combo = NoWheelComboBox()
        self.config_combo.setPlaceholderText("Choose a saved config...")
        self.config_combo.activated.connect(self._on_pick_config)
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self._on_save)
        config_row = QHBoxLayout()
        config_row.setContentsMargins(0, 0, 0, 0)
        config_row.addWidget(self.config_combo, 1)
        config_row.addWidget(self.save_btn)
        config_widget = QWidget()
        config_widget.setLayout(config_row)
        self._left_form.addRow("Saved config", config_widget)

        self.engine_combo = NoWheelComboBox()
        self.engine_combo.addItem("llama.cpp", "llama.cpp")
        self.engine_combo.addItem("ik_llama.cpp", "ik_llama.cpp")
        self._left_form.addRow("Engine", self.engine_combo)

        self.target_combo = NoWheelComboBox()
        self.target_combo.addItem("Native (build a binary)", "native")
        self.target_combo.addItem("Container (build an image)", "container")
        self._left_form.addRow("Target", self.target_combo)

        self.ref_edit = QLineEdit()
        self.ref_edit.setPlaceholderText("(engine default branch)")
        self.ref_edit.setToolTip("git ref (branch/tag/commit) to check out. "
                                  "Empty uses the engine's default branch.")
        self.ref_edit.textChanged.connect(self.refresh_preview)
        self._left_form.addRow("Git ref", self.ref_edit)

        self.source_dir_edit = QLineEdit()
        self.source_dir_edit.setToolTip(
            "Host checkout directory the native build runs against.")
        self.source_dir_edit.textChanged.connect(self.refresh_preview)
        self._left_form.addRow("Source dir", self.source_dir_edit)

        self.builder_image_edit = QLineEdit()
        self.builder_image_edit.setToolTip(
            "Multi-stage build image the Containerfile compiles in.")
        self.builder_image_edit.textChanged.connect(self.refresh_preview)
        self._left_form.addRow("Builder image", self.builder_image_edit)

        self.runtime_image_edit = QLineEdit()
        self.runtime_image_edit.setToolTip(
            "Slim runtime image the built binaries are copied into.")
        self.runtime_image_edit.textChanged.connect(self.refresh_preview)
        self._left_form.addRow("Runtime image", self.runtime_image_edit)

        self.tag_edit = QLineEdit()
        self.tag_edit.setPlaceholderText("(auto-generated)")
        self.tag_edit.setToolTip("Override the auto-generated image tag.")
        self.tag_edit.textChanged.connect(self.refresh_preview)
        self._left_form.addRow("Tag override", self.tag_edit)

        self.raw_defines_edit = QLineEdit()
        self.raw_defines_edit.setToolTip(
            "Extra -D defines appended verbatim, e.g. -DFOO=bar. "
            "Duplicates of catalog options are de-duplicated.")
        self.raw_defines_edit.textChanged.connect(self.refresh_preview)
        self._left_form.addRow("Raw defines", self.raw_defines_edit)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left)
        body.addWidget(left_scroll, 3)

        # RIGHT: catalog options, grouped, rebuilt whenever the engine flips
        # (the two repos' CMake catalogs genuinely differ, not just which
        # rows are visible -- see BUILD_CATALOG's per-setting `engine` gate).
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_inner = QWidget()
        self._right_layout = QVBoxLayout(right_inner)
        right_scroll.setWidget(right_inner)
        body.addWidget(right_scroll, 2)

        # BOTTOM: preview + copy actions + Generate
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        root.addWidget(self.preview)

        actions = QHBoxLayout()
        self.copy_configure_btn = QPushButton("Copy configure cmd")
        self.copy_build_btn = QPushButton("Copy build cmd")
        self.copy_containerfile_btn = QPushButton("Copy Containerfile")
        self.copy_podman_btn = QPushButton("Copy podman build cmd")
        self.copy_configure_btn.clicked.connect(
            lambda: self._copy(render_native(self.current_build_config()).configure_cmd))
        self.copy_build_btn.clicked.connect(
            lambda: self._copy(render_native(self.current_build_config()).build_cmd))
        self.copy_containerfile_btn.clicked.connect(
            lambda: self._copy(self._render_container().containerfile))
        self.copy_podman_btn.clicked.connect(
            lambda: self._copy(self._render_container().build_cmd))
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self.generate)
        for b in (self.copy_configure_btn, self.copy_build_btn,
                  self.copy_containerfile_btn, self.copy_podman_btn):
            actions.addWidget(b)
        actions.addStretch(1)
        actions.addWidget(self.generate_btn)
        root.addLayout(actions)

        # OUTPUTS: registry rows joined against `podman images` / on-disk
        # binaries, with a guarded delete.
        outputs_box = QGroupBox("Outputs")
        outputs_layout = QVBoxLayout(outputs_box)
        outputs_btns = QHBoxLayout()
        self.refresh_outputs_btn = QPushButton("Refresh outputs")
        self.refresh_outputs_btn.clicked.connect(self.refresh_outputs)
        self.delete_output_btn = QPushButton("Delete selected")
        self.delete_output_btn.clicked.connect(self.delete_selected_output)
        outputs_btns.addWidget(self.refresh_outputs_btn)
        outputs_btns.addWidget(self.delete_output_btn)
        outputs_btns.addStretch(1)
        outputs_layout.addLayout(outputs_btns)

        self.outputs_table = QTableWidget(0, 5)
        self.outputs_table.setHorizontalHeaderLabels(
            ["Identifier", "Status", "Size", "Created", "Config"])
        self.outputs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.outputs_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.outputs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.outputs_table.horizontalHeader().setStretchLastSection(True)
        outputs_layout.addWidget(self.outputs_table)
        root.addWidget(outputs_box)

        # Wire engine/target changes now that every field they touch exists.
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        self.target_combo.currentIndexChanged.connect(self._on_target_changed)
        self._on_engine_changed()
        self._on_target_changed()
        self._reload_config_combo()

    # -- settings form (rebuilt per engine) ----------------------------------

    def _rebuild_settings_form(self, engine: str) -> None:
        for box in self._group_boxes.values():
            box.setParent(None)
            box.deleteLater()
        self._group_boxes = {}
        self._widgets = {}
        groups: dict[str, QFormLayout] = {}
        for key, setting in for_engine(BUILD_CATALOG, engine).items():
            if setting.group not in groups:
                box = QGroupBox(setting.group)
                groups[setting.group] = QFormLayout(box)
                self._group_boxes[setting.group] = box
                self._right_layout.addWidget(box)
            w = make_widget(setting)
            w.changed.connect(self.refresh_preview)
            self._widgets[key] = w
            groups[setting.group].addRow(make_row_label(setting), w)
        if "cuda" in self._widgets:
            self._widgets["cuda"].changed.connect(self._on_cuda_toggle)

    def _on_engine_changed(self, _index=0) -> None:
        self._rebuild_settings_form(self.engine_combo.currentData() or "llama.cpp")
        self.refresh_preview()

    def _on_target_changed(self, _index=0) -> None:
        is_native = (self.target_combo.currentData() or "native") == "native"
        self._left_form.setRowVisible(self.source_dir_edit, is_native)
        self._left_form.setRowVisible(self.builder_image_edit, not is_native)
        self._left_form.setRowVisible(self.runtime_image_edit, not is_native)
        self._left_form.setRowVisible(self.tag_edit, not is_native)
        self.copy_configure_btn.setVisible(is_native)
        self.copy_build_btn.setVisible(is_native)
        self.copy_containerfile_btn.setVisible(not is_native)
        self.copy_podman_btn.setVisible(not is_native)
        if not is_native:
            self._maybe_seed_default_images()
        self.refresh_preview()

    def _on_cuda_toggle(self) -> None:
        cuda = self._widgets.get("cuda")
        arch = self._widgets.get("cuda-architectures")
        if cuda is not None and arch is not None and cuda.value() \
                and not str(arch.value()).strip():
            caps = query_compute_caps()
            if caps:
                arch.set_value(";".join(caps))
        self._maybe_seed_default_images()
        self.refresh_preview()

    def _maybe_seed_default_images(self) -> None:
        """Seed builder/runtime image fields from default_images(cfg) only
        when empty; never clobber a user-typed value. Same rule as
        ConfigurePanel._maybe_seed_default_image."""
        builder, runtime_img = default_images(self.current_build_config())
        if not self.builder_image_edit.text().strip():
            self.builder_image_edit.setText(builder)
        if not self.runtime_image_edit.text().strip():
            self.runtime_image_edit.setText(runtime_img)

    # -- saved configs --------------------------------------------------------

    def _reload_config_combo(self) -> None:
        configs = list_build_configs(self.base_dir)
        self._saved_configs = {c.name: c for c in configs}
        self.config_combo.blockSignals(True)
        self.config_combo.clear()
        for name in self._saved_configs:
            self.config_combo.addItem(name, name)
        self.config_combo.blockSignals(False)

    def _on_pick_config(self, _index=0) -> None:
        cfg = self._saved_configs.get(self.config_combo.currentData())
        if cfg is not None:
            self.load_build_config(cfg)

    def _on_save(self) -> None:
        save_build_config(self.current_build_config(), self.base_dir)
        self._reload_config_combo()

    # -- marshalling ------------------------------------------------------------

    def current_build_config(self) -> BuildConfig:
        options = {k: w.value() for k, w in self._widgets.items() if w.is_set()}
        return BuildConfig(
            name=self.name_edit.text().strip() or "build",
            engine=self.engine_combo.currentData() or "llama.cpp",
            target=self.target_combo.currentData() or "native",
            git_ref=self.ref_edit.text().strip(),
            source_dir=self.source_dir_edit.text().strip(),
            builder_image=self.builder_image_edit.text().strip(),
            runtime_image=self.runtime_image_edit.text().strip(),
            tag_override=self.tag_edit.text().strip(),
            options=options,
            raw_defines=self.raw_defines_edit.text().strip(),
        )

    def load_build_config(self, cfg: BuildConfig) -> None:
        self.name_edit.setText(cfg.name)
        self.engine_combo.blockSignals(True)
        idx = self.engine_combo.findData(cfg.engine)
        self.engine_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.engine_combo.blockSignals(False)
        self._rebuild_settings_form(cfg.engine)
        tidx = self.target_combo.findData(cfg.target)
        self.target_combo.setCurrentIndex(tidx if tidx >= 0 else 0)
        self.ref_edit.setText(cfg.git_ref)
        self.source_dir_edit.setText(cfg.source_dir)
        self.builder_image_edit.setText(cfg.builder_image)
        self.runtime_image_edit.setText(cfg.runtime_image)
        self.tag_edit.setText(cfg.tag_override)
        self.raw_defines_edit.setText(cfg.raw_defines)
        for key, w in self._widgets.items():
            w.set_value(w.setting.default)
            if key in cfg.options:
                w.set_value(cfg.options[key])
        self._on_target_changed()
        self.refresh_preview()

    # -- preview / generate -----------------------------------------------------

    def _containerfile_path(self, cfg: BuildConfig) -> Path:
        return builds_dir(self.base_dir) / f"{config_slug(cfg.name)}.containerfile"

    def _render_container(self, cfg: BuildConfig | None = None, tag: str | None = None):
        cfg = cfg or self.current_build_config()
        tag = tag or auto_tag(cfg, set(), datetime.date.today())
        return render_container(cfg, tag, str(self._containerfile_path(cfg)))

    def refresh_preview(self) -> None:
        cfg = self.current_build_config()
        if cfg.target == "container":
            cb = self._render_container(cfg)
            text = cb.containerfile + "\n" + cb.build_cmd
        else:
            nb = render_native(cfg)
            text = (nb.configure_cmd + "\n" + nb.build_cmd
                    + f"\n# expected binary: {nb.expected_binary}")
        self.preview.setPlainText(text)

    def generate(self) -> None:
        cfg = self.current_build_config()
        today = datetime.date.today()
        if cfg.target == "container":
            existing = {o.identifier for o in load_outputs(self.base_dir)}
            tag = auto_tag(cfg, existing, today)
            cb = self._render_container(cfg, tag)
            write_containerfile(cfg, cb.containerfile, self.base_dir)
            identifier, kind = tag, "tag"
        else:
            nb = render_native(cfg)
            identifier, kind = nb.expected_binary, "binary"
        add_output(BuildOutput(
            id=new_output_id(), kind=kind, identifier=identifier,
            config_name=cfg.name, engine=cfg.engine, git_ref=cfg.git_ref,
            options=cfg.options, created=today.isoformat(),
        ), self.base_dir)
        self.refresh_preview()

    @staticmethod
    def _copy(text: str) -> None:
        QApplication.clipboard().setText(text)

    # -- outputs table ----------------------------------------------------------

    @staticmethod
    def _binary_exists(path: str) -> bool:
        return Path(path).is_file()

    @staticmethod
    def _provenance_tooltip(output: BuildOutput | None) -> str:
        if output is None:
            return "Untracked image: no matching entry in the build registry."
        lines = [f"engine: {output.engine}", f"ref: {output.git_ref or '(default)'}"]
        non_default = {k: v for k, v in (output.options or {}).items()}
        if non_default:
            opts = ", ".join(f"{k}={v}" for k, v in non_default.items())
            lines.append(f"options: {opts}")
        if output.notes:
            lines.append(f"notes: {output.notes}")
        return "\n".join(lines)

    def refresh_outputs(self) -> None:
        """Threaded entry point: gather `podman images` off the UI thread,
        then finish (classify + render) on the UI thread via refresh_outputs_sync.
        """
        self._outputs_inflight = True
        QThreadPool.globalInstance().start(_OutputsGather(self, _BUILD_BINARY))
        QTimer.singleShot(150, self._poll_outputs)

    def _poll_outputs(self) -> None:
        if self._outputs_inflight:
            QTimer.singleShot(150, self._poll_outputs)
            return
        self.refresh_outputs_sync(images=self._outputs_images_result)

    def refresh_outputs_sync(self, images: dict | None = None) -> None:
        """Pure logic path, no thread pool: gather (or accept already-gathered)
        image metadata, classify every registered output against it, and fill
        outputs_table. Used directly by tests, and by _poll_outputs once the
        background `podman images` gather completes.
        """
        if images is None:
            images = list_images_detailed(_BUILD_BINARY)
        outputs = load_outputs(self.base_dir)
        rows = classify_outputs(outputs, images, self._binary_exists)
        untracked = untracked_custom_tags(images, outputs)
        for tag in untracked:
            rows.append(OutputRow(output=None, identifier=tag, status="untracked"))
        self._outputs_rows = rows

        table = self.outputs_table
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            created = row.created or (row.output.created if row.output else "")
            config_name = row.output.config_name if row.output else ""
            tooltip = self._provenance_tooltip(row.output)
            values = (row.identifier, row.status, row.size, created, config_name)
            for c, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(tooltip)
                table.setItem(r, c, item)

    def _confirm(self, text: str) -> bool:
        return QMessageBox.question(
            self, "Confirm", text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes

    def _error(self, text: str) -> None:
        QMessageBox.critical(self, "Error", text)

    def delete_selected_output(self) -> None:
        row_index = self.outputs_table.currentRow()
        if row_index < 0 or row_index >= len(self._outputs_rows):
            self._error("Select an output to delete first.")
            return
        row = self._outputs_rows[row_index]
        output = row.output
        identifier = row.identifier
        kind = output.kind if output is not None else "tag"

        using = profiles_using(identifier, kind, list_profiles(self.base_dir))
        if using:
            self._error(
                f"{identifier} is in use by profile(s): {', '.join(using)}. "
                "Repoint or delete those profiles first.")
            return

        if row.status == "built":
            if kind == "tag":
                if not self._confirm(f"Delete image {identifier}?"):
                    return
                ok, stderr = remove_image(_BUILD_BINARY, identifier)
                if not ok:
                    self._error(f"Failed to delete image {identifier}: {stderr}")
                    return
            else:
                build_dir = Path(identifier).parent.parent
                if not build_dir.name.startswith("build-"):
                    self._error(f"Refusing to delete non-build directory: {build_dir}")
                    return
                if not self._confirm(f"Delete build dir {build_dir}?"):
                    return
                try:
                    shutil.rmtree(build_dir)
                except OSError as exc:
                    self._error(f"Failed to delete build dir {build_dir}: {exc}")
                    return
            if output is not None:
                remove_output(output.id, self.base_dir)
            self.refresh_outputs_sync()
            return

        if row.status == "missing":
            if output is not None and self._confirm(
                    f"Remove {identifier} from the build registry?"):
                remove_output(output.id, self.base_dir)
                self.refresh_outputs_sync()
            return

        # "untracked": no registry entry to act on -- shown for awareness only.

    def use_in_profile(self, profile_name: str) -> None:
        """Set the selected output in the specified profile and save it.

        For tag outputs: sets profile.image
        For binary outputs: sets profile.runtime.native_binary
        """
        row_index = self.outputs_table.currentRow()
        if row_index < 0 or row_index >= len(self._outputs_rows):
            self._error("Select an output first.")
            return

        row = self._outputs_rows[row_index]
        output = row.output
        identifier = row.identifier
        kind = output.kind if output is not None else "tag"

        # Find the profile
        profiles = {p.name: p for p in list_profiles(self.base_dir)}
        if profile_name not in profiles:
            self._error(f"Profile {profile_name} not found.")
            return

        profile = profiles[profile_name]

        # Update the profile based on kind
        if kind == "tag":
            profile.image = identifier
        elif kind == "binary":
            profile.runtime = replace(profile.runtime, native_binary=identifier)
        else:
            self._error(f"Unknown output kind: {kind}")
            return

        # Save the profile
        save_profile(profile, self.base_dir)
