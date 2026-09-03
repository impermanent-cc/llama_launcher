"""The Build tab: a CMake build-option catalog form (native or container
target), a live command/Containerfile preview, and a "Generate" action that
renders the build via core/build_command.py, writes a Containerfile for the
container target, and records a BuildOutput in the store registry. This tab
never runs podman/cmake itself -- see core/build_command.py and
store/builds.py for the renderers and store that do the actual work.
"""

import datetime
import shutil
from pathlib import Path

from PySide6.QtCore import QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from llama_launcher.core.build_catalog import BUILD_CATALOG
from llama_launcher.core.build_command import (
    auto_tag,
    default_image_pool,
    default_images,
    render_container,
    render_native,
)
from llama_launcher.core.build_outputs import (
    OutputRow,
    classify_outputs,
    extract_build_dir,
    profiles_using,
    untracked_custom_tags,
)
from llama_launcher.core.build_spec import BuildConfig, BuildOutput
from llama_launcher.core.settings_catalog import for_engine
from llama_launcher.services.gpu import query_compute_caps
from llama_launcher.services.runtime import list_images_detailed, remove_image
from llama_launcher.store.builds import (
    add_output,
    containerfile_path,
    list_build_configs,
    load_outputs,
    new_output_id,
    remove_output,
    save_build_config,
    write_containerfile,
)
from llama_launcher.store.profiles import list_profiles, save_profile
from llama_launcher.ui.widgets.no_wheel import NoWheelComboBox
from llama_launcher.ui.widgets.setting_widgets import make_row_label, make_widget
from llama_launcher.ui.widgets.table_columns import set_resizable_columns


class _OutputsGather(QRunnable):
    """Off-UI-thread body of refresh_outputs(): the only slow part is the
    `podman images` subprocess call, so that's all this does. Same delivery
    model as _CapsGather below -- results live on the gather object itself
    and the panel keeps a reference to the CURRENT gather, so overlapping
    refreshes can't race: a newer refresh replaces the reference and every
    poll loop renders only the newest gather's result (a superseded gather's
    result is simply dropped). The classify+render work happens back on the
    UI thread in refresh_outputs_sync, called by _poll_outputs.
    """

    def __init__(self, binary: str):
        super().__init__()
        self._binary = binary
        self.done = False
        self.images: dict = {}

    def run(self):
        try:
            self.images = list_images_detailed(self._binary)
        except Exception:  # worker must never raise
            self.images = {}
        finally:
            self.done = True


class _WorkGather(QRunnable):
    """Off-UI-thread wrapper for a blocking (ok, error) callable -- used by
    delete_selected_output so `podman rmi` (up to 120s) and rmtree of a big
    build dir never freeze the window. Same done-flag poll shape as
    _CapsGather."""

    def __init__(self, fn):
        super().__init__()
        self._fn = fn
        self.done = False
        self.ok = False
        self.error = ""

    def run(self):
        try:
            self.ok, self.error = self._fn()
        except Exception as exc:  # worker must never raise
            self.ok, self.error = False, str(exc)
        finally:
            self.done = True


class _CapsGather(QRunnable):
    """Off-thread nvidia-smi compute-capability query for the CUDA-arch
    prefill. nvidia-smi can stall for seconds on a wedged driver or a waking
    dGPU; a synchronous call would freeze the whole window. Same
    owner-reference poll shape as _OutputsGather."""

    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.done = False
        self.caps: list = []

    def run(self):
        try:
            self.caps = query_compute_caps()
        finally:
            self.done = True


class BuildPanel(QWidget):
    # Emitted after use_in_profile saves a profile to disk, so MainWindow can
    # refresh its in-memory copy (a stale Configure form would silently
    # revert the change on its next Save).
    profile_updated = Signal(str)
    """Owns all Build-tab widgets. `base_dir` is the store directory (a
    profiles/builds root), passed explicitly rather than resolved internally
    so tests never touch the user's real config dir.
    """

    def __init__(self, *, base_dir, binary_provider=None, parent=None):
        super().__init__(parent)
        self.base_dir = Path(base_dir)
        # Which container binary the Outputs join/deletion talks to. The
        # Build tab has no Profile of its own, so the owner passes a provider
        # (MainWindow wires the Configure form's runtime choice); standalone
        # construction falls back to the same default Runtime.binary uses
        # everywhere else (core/spec.py).
        self._binary_provider = binary_provider or (lambda: "podman")
        self._widgets: dict[str, object] = {}
        self._group_boxes: dict[str, QGroupBox] = {}
        self._saved_configs: dict[str, BuildConfig] = {}
        self._outputs_rows: list[OutputRow] = []
        self._outputs_gather: _OutputsGather | None = None
        self._delete_gather: _WorkGather | None = None
        # In-memory copy of the outputs registry so per-keystroke previews
        # don't re-read outputs.json from disk; invalidated on every write.
        self._outputs_cache: list | None = None
        # Programmatic-load guard: seeding/prefill react to user gestures
        # only, never to load_build_config's own set_value cascade.
        self._loading = False
        # Option values carried across engine flips (the form is torn down
        # and rebuilt per engine); cleared when a saved config is loaded.
        self._options_stash: dict = {}
        self._caps_cache: list | None = None
        self._caps_gather: _CapsGather | None = None

        root = QVBoxLayout(self)

        # The form area and the two big bottom widgets share a draggable
        # vertical splitter; preview and outputs live in tabs so only one of
        # them takes height at a time.
        body_widget = QWidget()
        body = QHBoxLayout(body_widget)
        body.setContentsMargins(0, 0, 0, 0)

        # LEFT: config identity + engine/target + source/image fields
        left = QGroupBox("Build config")
        self._left_form = QFormLayout(left)
        self.name_edit = QLineEdit()
        self.name_edit.setToolTip(
            "Name for this build config; also the slug "
            "used for the saved-config file and any "
            "Containerfile it writes."
        )
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
        self.ref_edit.setToolTip(
            "git ref (branch/tag/commit) to check out. "
            "Empty uses the engine's default branch."
        )
        self.ref_edit.textChanged.connect(self.refresh_preview)
        self._left_form.addRow("Git ref", self.ref_edit)

        self.source_dir_edit = QLineEdit()
        self.source_dir_edit.setToolTip(
            "Host checkout directory the native build runs against."
        )
        self.source_dir_edit.textChanged.connect(self.refresh_preview)
        self._left_form.addRow("Source dir", self.source_dir_edit)

        self.builder_image_edit = QLineEdit()
        self.builder_image_edit.setToolTip(
            "Multi-stage build image the Containerfile compiles in."
        )
        self.builder_image_edit.textChanged.connect(self.refresh_preview)
        self._left_form.addRow("Builder image", self.builder_image_edit)

        self.runtime_image_edit = QLineEdit()
        self.runtime_image_edit.setToolTip(
            "Slim runtime image the built binaries are copied into."
        )
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
            "Duplicates of catalog options are de-duplicated."
        )
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

        # BOTTOM TAB 1: preview + copy actions + Generate
        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        preview_layout.addWidget(self.preview)

        actions = QHBoxLayout()
        self.copy_configure_btn = QPushButton("Copy configure cmd")
        self.copy_build_btn = QPushButton("Copy build cmd")
        self.copy_containerfile_btn = QPushButton("Copy Containerfile")
        self.copy_podman_btn = QPushButton("Copy podman build cmd")
        self.copy_configure_btn.clicked.connect(
            lambda: self._copy(render_native(self.current_build_config()).configure_cmd)
        )
        self.copy_build_btn.clicked.connect(
            lambda: self._copy(render_native(self.current_build_config()).build_cmd)
        )
        self.copy_containerfile_btn.clicked.connect(
            lambda: self._copy(self._render_container().containerfile)
        )
        self.copy_podman_btn.clicked.connect(
            lambda: self._copy(self._render_container().build_cmd)
        )
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self.generate)
        for b in (
            self.copy_configure_btn,
            self.copy_build_btn,
            self.copy_containerfile_btn,
            self.copy_podman_btn,
        ):
            actions.addWidget(b)
        actions.addStretch(1)
        actions.addWidget(self.generate_btn)
        preview_layout.addLayout(actions)

        # BOTTOM TAB 2: registry rows joined against `podman images` /
        # on-disk binaries, with a guarded delete. (Plain widget: the tab
        # label already says "Outputs", a group-box title would repeat it.)
        outputs_box = QWidget()
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
            ["Identifier", "Status", "Size", "Created", "Config"]
        )
        self.outputs_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.outputs_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.outputs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        set_resizable_columns(self.outputs_table, (240, 80, 80, 100, 120))
        self.outputs_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.outputs_table.customContextMenuRequested.connect(
            self._on_outputs_context_menu
        )
        outputs_layout.addWidget(self.outputs_table)

        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.addTab(preview_tab, "Command preview")
        self.bottom_tabs.addTab(outputs_box, "Outputs")

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(body_widget)
        splitter.addWidget(self.bottom_tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        # Wire engine/target changes now that every field they touch exists.
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        self.target_combo.currentIndexChanged.connect(self._on_target_changed)
        self._on_engine_changed()
        self._on_target_changed()
        self._reload_config_combo()

    # -- settings form (rebuilt per engine) ----------------------------------

    def _rebuild_settings_form(self, engine: str) -> None:
        # Stash the outgoing form's non-default values (and drop stash keys
        # the user reverted) so an engine flip -- which tears the widgets down
        # because the two catalogs genuinely differ -- doesn't silently lose
        # every option the user set. load_build_config clears the stash.
        for key, w in self._widgets.items():
            if w.is_set():
                self._options_stash[key] = w.value()
            else:
                self._options_stash.pop(key, None)
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
        # Re-apply stashed values that exist in this engine's catalog, as a
        # programmatic load (no seeding/prefill side effects).
        was_loading, self._loading = self._loading, True
        try:
            for key, value in self._options_stash.items():
                if key in self._widgets:
                    self._widgets[key].set_value(value)
        finally:
            self._loading = was_loading

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
        if self._loading:
            return
        self._maybe_prefill_cuda_arch()
        self._maybe_seed_default_images()
        self.refresh_preview()

    def _maybe_prefill_cuda_arch(self) -> None:
        """Fill an empty cuda-architectures field from the detected GPU. The
        first nvidia-smi query runs off-thread (it can stall for seconds);
        the result is cached, so later toggles are synchronous."""
        cuda = self._widgets.get("cuda")
        arch = self._widgets.get("cuda-architectures")
        if (
            cuda is None
            or arch is None
            or not cuda.value()
            or str(arch.value()).strip()
        ):
            return
        if self._caps_cache is not None:
            if self._caps_cache:
                arch.set_value(";".join(self._caps_cache))
            return
        if self._caps_gather is not None:
            return  # a fetch is already in flight
        g = _CapsGather(self)
        self._caps_gather = g
        QThreadPool.globalInstance().start(g)
        QTimer.singleShot(50, self._poll_caps)

    def _poll_caps(self) -> None:
        g = self._caps_gather
        if g is None:
            return
        if not g.done:
            QTimer.singleShot(50, self._poll_caps)
            return
        self._caps_gather = None
        self._caps_cache = g.caps
        self._maybe_prefill_cuda_arch()  # cache path; fills if still apt
        if g.caps:
            self.refresh_preview()

    def _maybe_seed_default_images(self) -> None:
        """Seed builder/runtime image fields from default_images(cfg) only
        when the field is empty or still holds ANY generator output (either
        the CUDA or the non-CUDA pair); never clobber genuinely user-typed
        text. Same value-set idiom as ConfigurePanel._maybe_seed_default_image
        -- without this, picking Container before ticking cuda seeds the
        debian pair and then the cuda branch of default_images() is
        unreachable, since the plain "only when empty" rule never re-seeds a
        field that already has generator text in it. Never fires during a
        programmatic load: a saved config that deliberately pairs pool values
        with a different cuda state must round-trip unmutated."""
        if self._loading:
            return
        builder_pool, runtime_pool = default_image_pool()
        builder, runtime_img = default_images(self.current_build_config())
        cur_builder = self.builder_image_edit.text().strip()
        if cur_builder == "" or cur_builder in builder_pool:
            self.builder_image_edit.setText(builder)
        cur_runtime = self.runtime_image_edit.text().strip()
        if cur_runtime == "" or cur_runtime in runtime_pool:
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
        try:
            save_build_config(self.current_build_config(), self.base_dir)
        except ValueError as e:  # reserved name ("outputs")
            self._error(str(e))
            return
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
        self._loading = True
        try:
            self._load_build_config(cfg)
        finally:
            self._loading = False
        self.refresh_preview()

    def _load_build_config(self, cfg: BuildConfig) -> None:
        self.name_edit.setText(cfg.name)
        self.engine_combo.blockSignals(True)
        idx = self.engine_combo.findData(cfg.engine)
        self.engine_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.engine_combo.blockSignals(False)
        self._rebuild_settings_form(cfg.engine)
        # Clear the stash AFTER the rebuild: a loaded config replaces prior UI
        # state, but _rebuild_settings_form re-stashes the OUTGOING form's set
        # values -- clearing first would let another engine's leftovers (e.g.
        # a llama-only cuda-fa=False under a loaded ik config) survive the
        # load and silently re-apply on the next manual engine flip.
        self._options_stash = {}
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

    # -- preview / generate -----------------------------------------------------

    def _containerfile_path(self, cfg: BuildConfig) -> Path:
        # Single source: store.builds owns the formula, so the written file
        # and the copyable `podman build -f` command can never diverge.
        return containerfile_path(cfg, self.base_dir)

    def _load_outputs_cached(self) -> list:
        """The outputs registry, from the in-memory copy when it's current.
        refresh_preview runs per keystroke; without this every keypress in
        container mode re-read outputs.json from disk."""
        if self._outputs_cache is None:
            self._outputs_cache = load_outputs(self.base_dir)
        return self._outputs_cache

    def _invalidate_outputs_cache(self) -> None:
        self._outputs_cache = None

    @staticmethod
    def _matching_entry(
        cfg: BuildConfig, kind: str, identifier: str | None, outputs: list
    ) -> BuildOutput | None:
        """The registry entry that describes the SAME expected build: for tags
        the same config/engine/ref/options snapshot (regenerating without
        changes reuses it instead of stacking phantom rows); for binaries the
        same path (one build dir holds one expected build)."""
        for o in outputs:
            if o.kind != kind:
                continue
            if kind == "binary":
                if o.identifier == identifier:
                    return o
            elif (
                o.config_name == cfg.name
                and o.engine == cfg.engine
                and o.git_ref == cfg.git_ref
                and o.options == cfg.options
            ):
                return o
        return None

    def _current_tag(self, cfg: BuildConfig, outs: list | None = None) -> str:
        """The tag `generate()` would record for `cfg` right now: the tag of
        an existing identical-build entry if one is registered (idempotent
        regenerate), else auto_tag() against the REAL existing-tags set from
        the registry. Shared by refresh_preview and generate so the
        previewed/copied tag and the recorded tag never diverge."""
        outs = outs if outs is not None else self._load_outputs_cached()
        dup = self._matching_entry(cfg, "tag", None, outs)
        if dup is not None:
            return dup.identifier
        existing = {o.identifier for o in outs}
        return auto_tag(cfg, existing, datetime.date.today())

    def _render_container(self, cfg: BuildConfig | None = None, tag: str | None = None):
        cfg = cfg or self.current_build_config()
        tag = tag or self._current_tag(cfg)
        return render_container(
            cfg,
            tag,
            str(self._containerfile_path(cfg)),
            binary=self._binary_provider() or "podman",
        )

    def refresh_preview(self) -> None:
        if self._loading:
            # load_build_config fires ~a dozen textChanged/changed cascades;
            # it performs the single real refresh after clearing the flag.
            return
        cfg = self.current_build_config()
        if cfg.target == "container":
            cb = self._render_container(cfg)
            text = cb.containerfile + "\n" + cb.build_cmd
        else:
            nb = render_native(cfg)
            text = (
                nb.configure_cmd
                + "\n"
                + nb.build_cmd
                + f"\n# expected binary: {nb.expected_binary}"
            )
        self.preview.setPlainText(text)

    def generate(self) -> None:
        cfg = self.current_build_config()
        today = datetime.date.today()
        outs = load_outputs(self.base_dir)  # authoritative read for a write
        if cfg.target == "container":
            tag = self._current_tag(cfg, outs)
            cb = self._render_container(cfg, tag)
            write_containerfile(cfg, cb.containerfile, self.base_dir)
            identifier, kind = tag, "tag"
        else:
            nb = render_native(cfg)
            identifier, kind = nb.expected_binary, "binary"
        # Re-clicking Generate must not stack registry rows: an identical
        # expected build keeps its entry; a binary regenerate with changed
        # flags/engine/ref REPLACES the entry for that build dir (the new
        # expectation supersedes the old one -- same path, one build). For
        # tags, _matching_entry already matched engine/ref/options, so the
        # entry is identical by construction; for binaries it matched on the
        # path alone, so engine/ref must be compared here or a ref change
        # would leave stale provenance in the registry forever.
        dup = self._matching_entry(cfg, kind, identifier, outs)
        if dup is not None:
            if kind == "tag" or (
                dup.options == cfg.options
                and dup.engine == cfg.engine
                and dup.git_ref == cfg.git_ref
            ):
                self.refresh_preview()
                return
            remove_output(dup.id, self.base_dir)
        add_output(
            BuildOutput(
                id=new_output_id(),
                kind=kind,
                identifier=identifier,
                config_name=cfg.name,
                engine=cfg.engine,
                git_ref=cfg.git_ref,
                options=cfg.options,
                created=today.isoformat(),
            ),
            self.base_dir,
        )
        self._invalidate_outputs_cache()
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
        if output.options:
            # options already holds only explicitly-set values (the form's
            # is_set filter); nothing to strip here.
            opts = ", ".join(f"{k}={v}" for k, v in output.options.items())
            lines.append(f"options: {opts}")
        if output.notes:
            lines.append(f"notes: {output.notes}")
        return "\n".join(lines)

    def refresh_outputs(self) -> None:
        """Threaded entry point: gather `podman images` off the UI thread,
        then finish (classify + render) on the UI thread via refresh_outputs_sync.
        A newer call supersedes an in-flight one: only the CURRENT gather's
        result is ever rendered, so overlapping refreshes (e.g. Refresh
        clicked mid-delete) can't paint a stale pre-delete snapshot last.
        """
        g = _OutputsGather(self._binary_provider() or "podman")
        self._outputs_gather = g
        QThreadPool.globalInstance().start(g)
        QTimer.singleShot(150, self._poll_outputs)

    def _poll_outputs(self) -> None:
        g = self._outputs_gather
        if g is None:
            return  # a newer poll chain already rendered
        if not g.done:
            QTimer.singleShot(150, self._poll_outputs)
            return
        self._outputs_gather = None
        self.refresh_outputs_sync(images=g.images)

    def refresh_outputs_sync(self, images: dict | None = None) -> None:
        """Pure logic path, no thread pool: gather (or accept already-gathered)
        image metadata, classify every registered output against it, and fill
        outputs_table. Used directly by tests, and by _poll_outputs once the
        background `podman images` gather completes.
        """
        if images is None:
            images = list_images_detailed(self._binary_provider() or "podman")
        self._invalidate_outputs_cache()  # pick up external registry edits
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
        return (
            QMessageBox.question(
                self,
                "Confirm",
                text,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            == QMessageBox.StandardButton.Yes
        )

    def _error(self, text: str) -> None:
        QMessageBox.critical(self, "Error", text)

    def _selected_output_row(self) -> OutputRow | None:
        """The OutputRow behind the table's current selection, or None."""
        row_index = self.outputs_table.currentRow()
        if row_index < 0 or row_index >= len(self._outputs_rows):
            return None
        return self._outputs_rows[row_index]

    @staticmethod
    def _row_kind(row: OutputRow) -> str:
        # Untracked rows have no registry entry; they are always images.
        return row.output.kind if row.output is not None else "tag"

    def delete_selected_output(self) -> None:
        row = self._selected_output_row()
        if row is None:
            self._error("Select an output to delete first.")
            return
        output = row.output
        identifier = row.identifier
        kind = self._row_kind(row)

        using = profiles_using(identifier, kind, list_profiles(self.base_dir))
        if using:
            self._error(
                f"{identifier} is in use by profile(s): {', '.join(using)}. "
                "Repoint or delete those profiles first."
            )
            return

        if row.status == "built":
            if kind == "tag":
                if not self._confirm(f"Delete image {identifier}?"):
                    return
                binary = self._binary_provider() or "podman"

                def work():
                    return remove_image(binary, identifier)

                fail_prefix = f"Failed to delete image {identifier}"
            else:
                # Same "which build dir owns this binary" rule the in-use
                # guard (profiles_using) applies -- never a second derivation.
                build_dir = Path(extract_build_dir(identifier) or identifier)
                if not build_dir.name.startswith("build-"):
                    self._error(f"Refusing to delete non-build directory: {build_dir}")
                    return
                if not self._confirm(f"Delete build dir {build_dir}?"):
                    return

                def work():
                    try:
                        shutil.rmtree(build_dir)
                    except OSError as exc:
                        return False, str(exc)
                    return True, ""

                fail_prefix = f"Failed to delete build dir {build_dir}"
            # `podman rmi` on a big layered image can run for minutes and
            # rmtree of a build dir isn't cheap either: run the blocking part
            # off the UI thread and finish in _poll_delete.
            if self._delete_gather is not None:
                return  # a delete is already in flight
            self.delete_output_btn.setEnabled(False)
            g = _WorkGather(work)
            g.fail_prefix = fail_prefix
            g.output_id = output.id if output is not None else None
            self._delete_gather = g
            QThreadPool.globalInstance().start(g)
            QTimer.singleShot(100, self._poll_delete)
            return

        if row.status == "missing":
            if output is not None and self._confirm(
                f"Remove {identifier} from the build registry?"
            ):
                remove_output(output.id, self.base_dir)
                self._invalidate_outputs_cache()
                self.refresh_outputs()
            return

        # "untracked": no registry entry to act on -- shown for awareness only.

    def _poll_delete(self) -> None:
        g = self._delete_gather
        if g is None:
            return
        if not g.done:
            QTimer.singleShot(100, self._poll_delete)
            return
        self._delete_gather = None
        self.delete_output_btn.setEnabled(True)
        if not g.ok:
            self._error(f"{g.fail_prefix}: {g.error}")
            return
        if g.output_id is not None:
            remove_output(g.output_id, self.base_dir)
            self._invalidate_outputs_cache()
        self.refresh_outputs()

    def _eligible_profiles(self, kind: str) -> list[str]:
        """List profiles eligible for the given output kind.

        For tag outputs: container-mode profiles (launch_mode == "container")
        For binary outputs: native-mode profiles (launch_mode == "native")
        """
        want = "container" if kind == "tag" else "native"
        return [
            p.name
            for p in list_profiles(self.base_dir)
            if p.runtime.launch_mode == want
        ]

    def _on_outputs_context_menu(self, pos) -> None:
        """Show a context menu on the outputs table with "Use in profile" actions."""
        row_index = self.outputs_table.rowAt(pos.y())
        if row_index < 0 or row_index >= len(self._outputs_rows):
            return
        # (The right-click press already moved currentRow here, so
        # use_in_profile's selection-based lookup acts on this same row.)
        kind = self._row_kind(self._outputs_rows[row_index])

        eligible = self._eligible_profiles(kind)
        if not eligible:
            return

        menu = QMenu()
        for profile_name in eligible:
            action = menu.addAction(f"Use in profile: {profile_name}")
            action.triggered.connect(
                lambda checked=False, name=profile_name: self.use_in_profile(name)
            )

        menu.exec(self.outputs_table.viewport().mapToGlobal(pos))

    def use_in_profile(self, profile_name: str) -> None:
        """Set the selected output in the specified profile and save it.

        For tag outputs: sets profile.image
        For binary outputs: sets profile.runtime.native_binary
        """
        row = self._selected_output_row()
        if row is None:
            self._error("Select an output first.")
            return
        identifier = row.identifier
        kind = self._row_kind(row)

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
            profile.runtime.native_binary = identifier
        else:
            self._error(f"Unknown output kind: {kind}")
            return

        # Save the profile and tell MainWindow, whose in-memory copy (and
        # possibly the loaded Configure form) is now stale.
        save_profile(profile, self.base_dir)
        self.profile_updated.emit(profile_name)
