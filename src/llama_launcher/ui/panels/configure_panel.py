import os
import posixpath

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QCheckBox,
    QGroupBox, QScrollArea, QLabel, QPlainTextEdit, QPushButton,
    QMessageBox, QFileDialog, QInputDialog,
)

from llama_launcher.core.capabilities import (
    describe_relevance, Tier, suggestions as compute_suggestions,
)
from llama_launcher.core.command_builder import build_command
from llama_launcher.core.pathmap import host_to_container, container_to_host
from llama_launcher.core.settings_catalog import (
    CATALOG, member_catalog, router_catalog, for_engine,
    KV_CACHE_TYPES, IK_EXTRA_KV_CACHE_TYPES,
)
from llama_launcher.core.spec import (
    DEFAULT_STOP_TIMEOUT, Profile, Runtime, RouterMember, member_model_id,
)
from llama_launcher.core.validation import validate, Issue
from llama_launcher.services import api_key as api_key_store
from llama_launcher.services import model_info, runtime, native
from llama_launcher.store.profiles import list_profiles, resolve_member_pairs
from llama_launcher.ui.widgets.setting_widgets import make_widget, SuggestionDot
from llama_launcher.ui.widgets.no_wheel import NoWheelComboBox, NoWheelSpinBox
from llama_launcher.ui.panels.mounts_panel import MountsPanel
from llama_launcher.ui.panels.lora_panel import LoraPanel
from llama_launcher.ui.widgets.collapsible import CollapsibleSection
from llama_launcher.ui.widgets.api_key_box import ApiKeyBox
from llama_launcher.ui.widgets.harness_info_box import HarnessInfoBox
from llama_launcher.ui.widgets.status_banner import StatusBanner


_ENGINE_DEFAULT_IMAGE = {
    "llama.cpp": "ghcr.io/ggml-org/llama.cpp:server-cuda",
    "ik_llama.cpp": "ghcr.io/ikawrakow/ik-llama-cpp:cu12-server",
}


class ConfigurePanel(QWidget):
    """The Configure tab body: environment fields, members, settings form,
    command preview + api-key box. Owns all Configure-tab widgets. Holds a
    back-reference to the MainWindow for the few cross-panel calls it makes.
    """

    def __init__(self, window, *, parent=None):
        super().__init__(parent)
        self.window = window
        self.configure_tab = self
        self._profile = Profile(name="New Profile")
        self._last_caps = None

        body = QHBoxLayout(self)
        # Trim the body's own margins so the Environment/Settings columns get a
        # little extra vertical room (the bottom command-preview strip is kept
        # compact rather than made draggable).
        body.setContentsMargins(4, 4, 4, 2)

        # LEFT: environment (image + model only for v1 binding; mounts editor TODO-UI)
        left = QGroupBox("Environment")
        left_form = QFormLayout(left)
        self._left_form = left_form
        self.image_edit = QLineEdit()
        self.model_edit = QLineEdit()
        self.binary_combo = NoWheelComboBox(); self.binary_combo.addItems(["podman", "docker"])
        self.engine_combo = NoWheelComboBox()
        self.engine_combo.addItem("llama.cpp", "llama.cpp")
        self.engine_combo.addItem("ik_llama.cpp", "ik_llama.cpp")
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        self.gpu_combo = NoWheelComboBox()
        for _gpu_label, _gpu_val in (
            ("CDI — --device nvidia.com/gpu=all (recommended)", "cdi"),
            ("Legacy — --gpus all", "gpus-all"),
            ("None — no GPU passthrough", "none"),
        ):
            self.gpu_combo.addItem(_gpu_label, _gpu_val)
        for w in (self.image_edit, self.model_edit):
            w.textChanged.connect(self.refresh_preview)
        self.binary_combo.currentTextChanged.connect(self.refresh_preview)
        self.gpu_combo.currentTextChanged.connect(self.refresh_preview)
        self.update_badge = QPushButton("")
        self.update_badge.setFlat(True)
        self.update_badge.setVisible(False)
        self.update_badge.clicked.connect(self.window._launch.on_fetch_latest)
        self.detect_image_btn = QPushButton("Detect")
        self.detect_image_btn.setToolTip(
            "Fill the Image field from llama.cpp images already pulled locally "
            "(podman/docker images).")
        self.detect_image_btn.clicked.connect(self.window._launch.detect_image)
        self.fetch_btn = QPushButton("Fetch latest")
        self.fetch_btn.setToolTip(
            "Query the container registry (GHCR) for the newest build tag matching "
            "this image's repo and variant, and update the Image field. Requires an "
            "image to be set (use Detect or type one). This updates the tag only — it "
            "does NOT download the build; pull it with podman/docker pull.")
        self.fetch_btn.clicked.connect(self.window._launch.on_fetch_latest)
        image_row = QHBoxLayout()
        image_row.setContentsMargins(0, 0, 0, 0)
        image_row.addWidget(self.image_edit, 1)
        image_row.addWidget(self.detect_image_btn)
        image_row.addWidget(self.fetch_btn)
        image_row.addWidget(self.update_badge)
        image_widget = QWidget()
        image_widget.setLayout(image_row)
        self._image_row = image_widget   # container-only row, hidden in native mode
        from PySide6.QtWidgets import QTableWidget, QHeaderView

        # NoWheel: an unfocused scroll over these must not silently flip launch
        # semantics (terminal vs detached) or expose the port to the network.
        self.mode_combo = NoWheelComboBox()
        self.mode_combo.addItem("Single server", "server")
        self.mode_combo.addItem("Router (headless host)", "router")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self.bind_host_combo = NoWheelComboBox()
        self.bind_host_combo.setEditable(True)
        self.bind_host_combo.addItems(["127.0.0.1", "0.0.0.0"])
        self.bind_host_combo.currentTextChanged.connect(self.refresh_preview)

        # Launch mode: container (podman/docker, default) vs native (a
        # directly-run prebuilt llama-server binary). NoWheel for the same
        # reason as mode_combo -- an unfocused scroll must not silently swap
        # how the server is launched.
        self.launch_mode_combo = NoWheelComboBox()
        self.launch_mode_combo.addItem("Container (podman/docker)", "container")
        self.launch_mode_combo.addItem("Native (run a built binary)", "native")
        self.launch_mode_combo.currentIndexChanged.connect(self._on_launch_mode_changed)

        self.native_binary_edit = QLineEdit()
        self.native_binary_edit.setPlaceholderText("/path/to/llama-server")
        self.native_binary_edit.setToolTip(
            "Path to a prebuilt llama-server executable (mainline or ik_llama.cpp). "
            "The launcher runs it directly as a managed background process.")
        self.native_binary_edit.textChanged.connect(self.refresh_preview)

        # `podman stop -t` grace period for the Stop button. Large MoE models can
        # need more than podman's 10s default to unload cleanly before SIGKILL.
        self.stop_timeout_spin = NoWheelSpinBox()
        self.stop_timeout_spin.setRange(1, 300)
        self.stop_timeout_spin.setValue(DEFAULT_STOP_TIMEOUT)
        self.stop_timeout_spin.setSuffix(" s")
        self.stop_timeout_spin.setToolTip(
            "Grace period after Stop before the container is force-killed "
            "(SIGTERM → wait → SIGKILL). Raise it if a large model needs longer "
            "to unload cleanly.\n\nApplies to this profile's own container — in "
            "router mode that's the router container; each router member's kill "
            "delay is the per-row 'Stop timeout (s)' in the Router members table.")

        # A table, not a list: model id / load-on-startup / stop-timeout are all
        # per-member settings the spec requires to be editable, and inline
        # editing keeps them reachable without a modal dialog (which would hang
        # the headless test run).
        self.members_list = QTableWidget(0, 4)
        self.members_list.setHorizontalHeaderLabels(
            ["Profile", "Model id (harness)", "Load at start", "Stop timeout (s)"])
        for _col, _tip in enumerate((
            "A saved model profile to serve. Configure its GPU layers / MoE / "
            "context in that profile (single-server mode), then add it here.",
            "Name the harness calls this model. Empty = derived from the profile name.",
            "Load this member as soon as the router starts (otherwise on first request).",
            "Seconds to wait after unload before force-killing this member's container.",
        )):
            item = self.members_list.horizontalHeaderItem(_col)
            if item is not None:
                item.setToolTip(_tip)
        self.members_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.members_list.verticalHeader().setVisible(False)
        self.members_list.setMaximumHeight(140)
        self.members_list.itemChanged.connect(lambda _i: self.refresh_preview())
        self.add_member_btn = QPushButton("Add member…")
        self.add_member_btn.clicked.connect(self._on_add_member)
        self.remove_member_btn = QPushButton("Remove member")
        self.remove_member_btn.clicked.connect(self._on_remove_member)
        self.edit_member_btn = QPushButton("Edit member…")
        self.edit_member_btn.setToolTip(
            "Load the selected member's profile into the form to set its GPU "
            "layers, MoE offload, context, etc. (double-clicking a row does the same).")
        self.edit_member_btn.clicked.connect(self._on_edit_member)
        members_row = QHBoxLayout()
        members_row.setContentsMargins(0, 0, 0, 0)
        members_row.addWidget(self.add_member_btn)
        members_row.addWidget(self.edit_member_btn)
        members_row.addWidget(self.remove_member_btn)
        members_widget = QWidget()
        self._members_row = members_widget
        members_box = QVBoxLayout(members_widget)
        members_box.setContentsMargins(0, 0, 0, 0)
        members_box.addWidget(self.members_list)
        members_box.addLayout(members_row)
        self.members_guidance = QLabel(
            "Each member is a saved model profile — set its GPU layers, MoE offload, "
            "and context in that profile (single-server mode), then add it here.")
        self.members_guidance.setWordWrap(True)
        self.members_guidance.setStyleSheet("QLabel { color: palette(mid); }")
        members_box.addWidget(self.members_guidance)
        self.members_list.cellDoubleClicked.connect(
            lambda _r, _c: self._on_edit_member() if _c == 0 else None)

        left_form.addRow("Launch mode", self.launch_mode_combo)
        left_form.addRow("Mode", self.mode_combo)
        left_form.addRow("Bind address", self.bind_host_combo)
        left_form.addRow("Stop grace period", self.stop_timeout_spin)
        left_form.addRow("Router members", members_widget)

        left_form.addRow("Engine", self.engine_combo)
        left_form.addRow("Image", image_widget)
        left_form.addRow("Model", self._field_with_browse(self.model_edit))
        left_form.addRow("Runtime", self.binary_combo)
        self._native_binary_row = self._native_binary_field_with_browse()
        left_form.addRow("llama-server binary", self._native_binary_row)
        left_form.addRow("GPU", self.gpu_combo)

        # mounts editor
        self.mounts_panel = MountsPanel()
        self.mounts_panel.changed.connect(self.refresh_preview)
        left_form.addRow("Folders", self.mounts_panel)
        self.mmproj_edit = QLineEdit(); self.mmproj_edit.textChanged.connect(self.refresh_preview)
        self.draft_model_edit = QLineEdit(); self.draft_model_edit.textChanged.connect(self.refresh_preview)
        self.raw_edit = QLineEdit(); self.raw_edit.textChanged.connect(self.refresh_preview)
        self._mmproj_dot = SuggestionDot(self)
        self._draft_model_dot = SuggestionDot(self)
        left_form.addRow("mmproj", self._field_with_browse(self.mmproj_edit, self._mmproj_dot))
        left_form.addRow("draft model", self._field_with_browse(self.draft_model_edit, self._draft_model_dot))
        self.lora_panel = LoraPanel()
        self.lora_panel.changed.connect(self.refresh_preview)
        self.lora_section = CollapsibleSection("LoRA adapters", self.lora_panel, collapsed=True)
        left_form.addRow(self.lora_section)
        left_form.addRow("Raw args", self.raw_edit)
        self.extra_args_edit = QLineEdit()
        self.extra_args_edit.textChanged.connect(self.refresh_preview)
        left_form.addRow("Extra podman args", self.extra_args_edit)
        self.selinux_check = QCheckBox("Disable SELinux labels (--security-opt=label=disable)")
        self.selinux_check.toggled.connect(self.refresh_preview)
        left_form.addRow(self.selinux_check)

        # Set tooltips on Environment field widgets
        self.mode_combo.setToolTip(
            "Single server runs one model. Router (headless host) serves several "
            "member models on one port with an API key, loading them on demand.")
        self.bind_host_combo.setToolTip(
            "Address the server binds to. 127.0.0.1 = this machine only; "
            "0.0.0.0 = reachable from the network (an API key is then required).")
        self.image_edit.setToolTip(
            "Container image for llama-server, e.g. "
            "ghcr.io/ggml-org/llama.cpp:server-cuda. Use Detect to list local images.")
        self.model_edit.setToolTip(
            "Path to the .gguf model as seen INSIDE the container, e.g. "
            "/models/Qwen3-A3B-Q4.gguf (add a Folder that maps the host dir).")
        self.binary_combo.setToolTip("Container runtime used to launch the server.")
        self.engine_combo.setToolTip(
            "Which llama.cpp-family server to run. 'llama.cpp' = mainline "
            "(ghcr.io/ggml-org/llama.cpp); 'ik_llama.cpp' = ikawrakow's fork with "
            "extra MoE/quant flags (ghcr.io/ikawrakow/ik-llama-cpp).")
        self.gpu_combo.setToolTip(
            "GPU passthrough. CDI (nvidia.com/gpu=all) is recommended on modern "
            "NVIDIA + podman; Legacy uses --gpus all; None runs CPU-only.")
        self.mmproj_edit.setToolTip(
            "Optional multimodal projector .gguf (container path) for vision models.")
        self.draft_model_edit.setToolTip(
            "Optional small draft model .gguf (container path) for speculative decoding.")
        self.raw_edit.setToolTip(
            "Extra llama-server flags appended verbatim, e.g. --temp 0.6 --top-k 20. "
            "Duplicates of structured settings are de-duplicated.")
        self.extra_args_edit.setToolTip(
            "Extra flags for the container runtime (podman/docker) itself, e.g. "
            "--shm-size=1g. Not passed to llama-server.")
        self.selinux_check.setToolTip(
            "Add --security-opt=label=disable. Needed on some SELinux hosts when a "
            "mounted model dir is otherwise unreadable in the container.")
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left)
        body.addWidget(left_scroll, 3)

        # RIGHT: settings grouped, scrollable
        self._widgets: dict[str, object] = {}
        right_scroll = QScrollArea(); right_scroll.setWidgetResizable(True)
        right_inner = QWidget(); right_layout = QVBoxLayout(right_inner)
        groups: dict[str, QFormLayout] = {}
        self._group_boxes: dict[str, QGroupBox] = {}
        # key -> (form layout, widget), so _on_mode_changed can hide whole rows
        # (label included) for settings that don't apply to the current mode.
        self._setting_rows: dict[str, tuple] = {}
        for key, setting in CATALOG.items():
            if setting.group not in groups:
                box = QGroupBox(setting.group)
                groups[setting.group] = QFormLayout(box)
                self._group_boxes[setting.group] = box
                right_layout.addWidget(box)
            w = make_widget(setting)
            w.changed.connect(self.refresh_preview)
            self._widgets[key] = w
            groups[setting.group].addRow(setting.flag, w)
            self._setting_rows[key] = (groups[setting.group], w)
        # When load-mode is active it supersedes --no-mmap/--mlock (see
        # command_builder), so gray those legacy checkboxes out to signal they're
        # ignored. Kept in sync on edit, mode change and profile load.
        if "load-mode" in self._widgets:
            self._widgets["load-mode"].changed.connect(self._sync_load_mode_legacy)
        right_scroll.setWidget(right_inner)
        body.addWidget(right_scroll, 2)

        # BOTTOM: command preview is config-only; wrap it in one container so
        # it can be hidden on the Monitor/Router/Benchmark tabs (see
        # _on_tab_changed). Launch/Stop/etc stay shared below.
        self._config_bottom = QWidget()
        config_bottom_box = QVBoxLayout(self._config_bottom)
        config_bottom_box.setContentsMargins(0, 0, 0, 0)
        config_bottom_box.setSpacing(3)   # tighten the bottom strip's dead space
        self.model_meta_label = QLabel("")
        config_bottom_box.addWidget(self.model_meta_label)
        self.model_edit.textChanged.connect(lambda _: self.apply_model_caps())
        self.mounts_panel.changed.connect(self.apply_model_caps)
        self.api_key_box = ApiKeyBox()
        self.api_key_box.key_scope_changed.connect(self.window._on_key_scope_changed)
        self.api_key_box.key_saved.connect(self.window._on_key_saved)
        self.harness_box = HarnessInfoBox()
        self.configure_status = StatusBanner()
        config_bottom_box.addWidget(self.configure_status)
        config_bottom_box.addWidget(self.api_key_box)
        config_bottom_box.addWidget(self.harness_box)
        preview_row = QHBoxLayout()
        self.preview = QPlainTextEdit(); self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(90)
        preview_row.addWidget(self.preview, 1)
        self.export_sh_btn = QPushButton("Export .sh")
        self.export_sh_btn.clicked.connect(self.window._report._on_export_sh)
        config_bottom_box.addWidget(QLabel("Command preview:"))
        config_bottom_box.addLayout(preview_row)

        # lifecycle / launch buttons (added to a shared row in MainWindow, below
        # the tab widget, so they're visible regardless of the active tab)
        self.launch_btn = QPushButton("▶ Launch")
        self.stop_btn = QPushButton("■ Stop")
        self.restart_btn = QPushButton("⟳ Restart")
        self.web_ui_btn = QPushButton("Open Web UI")
        self.web_ui_btn.setEnabled(False)
        self.web_ui_btn.clicked.connect(self.window._report.open_web_ui)
        self.detached_check = QCheckBox("Run detached (no terminal window)")
        self.detached_check.setToolTip(
            "Launch without a terminal window; watch output on the Monitor "
            "tab and use Stop to shut it down.")
        self.detached_check.toggled.connect(self.refresh_preview)

        # profile bar widgets (assembled into the top bar, alongside
        # MainWindow-owned widgets, in MainWindow.__init__)
        self.name_edit = QLineEdit(self._profile.name)
        self.name_edit.setPlaceholderText("Profile name")
        self.name_edit.setToolTip(
            "Name for this profile. Also sets the container name "
            "(llama-<name>), so pick something filesystem-friendly.")
        self.name_edit.textChanged.connect(self.refresh_preview)
        self.profile_combo = NoWheelComboBox()
        self.save_btn = QPushButton("Save")
        self.save_as_btn = QPushButton("Save As")
        self.delete_btn = QPushButton("Delete")
        self.report_btn = QPushButton("Generate report")
        self.save_btn.clicked.connect(self.window.save_current_profile)
        self.save_as_btn.clicked.connect(self.window.save_as_profile)
        self.delete_btn.clicked.connect(self.window.delete_current_profile)
        self.report_btn.clicked.connect(self.window._report.on_generate_report)
        self.profile_combo.activated.connect(self.window._on_pick_profile)

        # lifecycle buttons
        self.launch_btn.clicked.connect(self.window._launch.on_launch)
        self.stop_btn.clicked.connect(self.window._launch.on_stop)
        self.restart_btn.clicked.connect(self.window._launch.on_restart)

    # -- Configure marshalling / handlers (moved from MainWindow) -----------

    def _profile_name(self) -> str:
        """Current profile name from the Name field, falling back to a default
        so the container name / preview are never empty."""
        return self.name_edit.text().strip() or "New Profile"

    def active_catalog(self) -> dict:
        """The settings that apply to the mode currently selected in the form.

        Router CLI args outrank every member's preset value, so a router must
        not be able to carry model-level settings; conversely --models-max on a
        single-model llama-server is rejected outright.
        """
        base = (router_catalog() if self.mode_combo.currentData() == "router"
                else member_catalog())
        return for_engine(base, self.engine_combo.currentData() or "llama.cpp")

    def _apply_mode_to_settings_form(self) -> None:
        active = self.active_catalog()
        for group, box in self._group_boxes.items():
            box.setVisible(False)
        for key, (form, widget) in self._setting_rows.items():
            visible = key in active
            index = form.getWidgetPosition(widget)[0]
            if index >= 0:
                form.setRowVisible(index, visible)
            if visible:
                self._group_boxes[CATALOG[key].group].setVisible(True)

    def _on_mode_changed(self, _index=0) -> None:
        is_router = self.mode_combo.currentData() == "router"
        # Hide the whole "Router members" form row (its QFormLayout label too),
        # not just the inner widgets -- otherwise an orphaned "Router members"
        # label lingers on the server-mode form.
        self._left_form.setRowVisible(self._members_row, is_router)
        self._apply_mode_to_settings_form()
        # A router has no model of its own; its members carry those fields.
        for w in (self.model_edit, self.mmproj_edit, self.draft_model_edit):
            w.setEnabled(not is_router)
        self.lora_panel.setEnabled(not is_router)
        self._update_detached_visibility()
        # The reusable API key + harness block + status/exposure banner are
        # ROUTER-only concepts (a single server uses its own --api-key field in
        # the Settings column). Show them only in router mode.
        self.api_key_box.setVisible(is_router)
        self.harness_box.setVisible(is_router)
        self.configure_status.setVisible(is_router)
        self.window.monitor_status.setVisible(is_router)
        self._sync_load_mode_legacy()
        self.refresh_preview()
        if is_router:
            self.window.refresh_router_panel_header()

    def _update_detached_visibility(self) -> None:
        """"Run detached" only makes sense for a container launch in server
        mode: a router is never detached (it's always a managed background
        process) and neither is native mode (native always runs as a managed
        background subprocess -- there's no terminal-vs-detached choice)."""
        is_router = self.mode_combo.currentData() == "router"
        is_native = self.launch_mode_combo.currentData() == "native"
        self.detached_check.setVisible(not is_router and not is_native)

    def _on_launch_mode_changed(self, *_) -> None:
        native = self.launch_mode_combo.currentData() == "native"
        # Native shows the binary path; hides everything container-only.
        self._left_form.setRowVisible(self._native_binary_row, native)
        self._left_form.setRowVisible(self._image_row, not native)
        self._left_form.setRowVisible(self.binary_combo, not native)
        self._left_form.setRowVisible(self.gpu_combo, not native)
        self._left_form.setRowVisible(self.mounts_panel, not native)
        self._update_detached_visibility()
        self.refresh_preview()

    def _apply_engine_enums(self) -> None:
        """Extend/revert -ctk/-ctv enum choices for the current engine."""
        engine = self.engine_combo.currentData() or "llama.cpp"
        extra = IK_EXTRA_KV_CACHE_TYPES if engine == "ik_llama.cpp" else ()
        for k in ("cache-type-k", "cache-type-v"):
            w = self._widgets.get(k)
            if w is not None:
                w.set_enum_choices(tuple(KV_CACHE_TYPES) + tuple(extra))

    def _maybe_seed_default_image(self, engine: str) -> None:
        """Seed a sensible default image only when the field is empty or still
        holds the OTHER engine's default; never clobber a user-typed value."""
        cur = self.image_edit.text().strip()
        if cur == "" or cur in set(_ENGINE_DEFAULT_IMAGE.values()):
            self.image_edit.setText(_ENGINE_DEFAULT_IMAGE[engine])

    def _on_engine_changed(self, _index=0) -> None:
        engine = self.engine_combo.currentData() or "llama.cpp"
        self._apply_engine_enums()
        self._apply_mode_to_settings_form()   # show/hide the ik group by active_catalog
        self._maybe_seed_default_image(engine)
        self.refresh_preview()

    def _sync_load_mode_legacy(self) -> None:
        """Gray out --no-mmap/--mlock when load-mode is set (it wins in argv)."""
        lm = self._widgets.get("load-mode")
        if lm is None:
            return
        load_mode_active = lm.value() != CATALOG["load-mode"].default
        for key in ("no-mmap", "mlock"):
            w = self._widgets.get(key)
            if w is not None:
                w.setEnabled(not load_mode_active)

    def _add_member_item(self, member: RouterMember) -> None:
        from PySide6.QtWidgets import QTableWidgetItem
        row = self.members_list.rowCount()
        # itemChanged fires per setItem; without this the handler reads a
        # half-populated row and hits None cells.
        blocked = self.members_list.blockSignals(True)
        self.members_list.insertRow(row)

        name = QTableWidgetItem(member.profile)
        name.setFlags(name.flags() & ~Qt.ItemIsEditable)   # the profile is the identity
        self.members_list.setItem(row, 0, name)

        # Empty means "derive from the profile name"; show the derived value as a
        # placeholder so the harness-facing id is never a mystery.
        mid = QTableWidgetItem(member.model_id)
        mid.setToolTip(f"Empty = {member_model_id(member)}")
        self.members_list.setItem(row, 1, mid)

        load = QTableWidgetItem()
        load.setFlags((load.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable)
        load.setCheckState(Qt.Checked if member.load_on_startup else Qt.Unchecked)
        self.members_list.setItem(row, 2, load)

        self.members_list.setItem(row, 3, QTableWidgetItem(str(member.stop_timeout)))
        self.members_list.blockSignals(blocked)

    def set_member_fields(self, row: int, model_id: str | None = None,
                          load_on_startup: bool | None = None,
                          stop_timeout: int | None = None) -> None:
        """Programmatic equivalent of editing a member row (used by tests)."""
        if model_id is not None:
            self.members_list.item(row, 1).setText(model_id)
        if load_on_startup is not None:
            self.members_list.item(row, 2).setCheckState(
                Qt.Checked if load_on_startup else Qt.Unchecked)
        if stop_timeout is not None:
            self.members_list.item(row, 3).setText(str(stop_timeout))
        self.refresh_preview()

    def _member_candidates(self) -> list[str]:
        """Non-router profiles eligible to be added as router members.

        Read fresh from disk so a profile saved this session shows up without a
        restart. A router can't be a member, so router-mode profiles are
        filtered out -- that filter also excludes the router being edited once
        it's saved. We deliberately do NOT exclude by the Name field's current
        text: the natural add-a-member flow (switch to server mode, save the new
        model, switch back to router mode) leaves the new model's name in the
        Name field, and excluding it hid the very member the user just made until
        an app restart.
        """
        return [p.name for p in list_profiles(self.window.router_base_dir()) if p.mode != "router"]

    def _on_add_member(self) -> None:
        names = self._member_candidates()
        if not names:
            QMessageBox.information(self, "No profiles",
                                    "Save a model profile first; routers serve members.")
            return
        name, ok = QInputDialog.getItem(self, "Add member", "Profile:", names, 0, False)
        if ok and name:
            self._add_member_item(RouterMember(profile=name))
            self.refresh_preview()

    def _on_remove_member(self) -> None:
        for row in sorted({i.row() for i in self.members_list.selectedIndexes()},
                          reverse=True):
            self.members_list.removeRow(row)
        self.refresh_preview()

    def _has_unsaved_changes(self) -> bool:
        # Stateless dirty check: the form is clean iff current_profile() round-trips
        # equal to the stored profile of the same name (verified: dataclass equality,
        # dict-order-independent). A never-saved profile counts as changed.
        cur = self.current_profile()
        saved = {p.name: p for p in list_profiles(self.window.router_base_dir())}.get(cur.name)
        return saved is None or cur != saved

    def _on_edit_member(self) -> None:
        row = self.members_list.currentRow()
        if row < 0:
            return
        item = self.members_list.item(row, 0)   # column 0 = member profile name
        if item is None:
            return
        name = item.text()
        target = {p.name: p for p in list_profiles(self.window.router_base_dir())}.get(name)
        if target is None:
            QMessageBox.warning(
                self, "Profile missing",
                f"Profile '{name}' no longer exists — remove it from the router "
                f"or recreate it.")
            return
        if self._has_unsaved_changes():
            choice = QMessageBox.question(
                self, "Unsaved changes",
                f"You have unsaved changes to '{self.current_profile().name}'. "
                f"Editing member '{name}' will load its profile and lose those changes.",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Cancel)
            if choice == QMessageBox.Cancel:
                return
            if choice == QMessageBox.Save:
                self.window.save_current_profile()
        self.load_profile(target)

    def members(self) -> list:
        out = []
        for row in range(self.members_list.rowCount()):
            name = self.members_list.item(row, 0)
            mid = self.members_list.item(row, 1)
            load = self.members_list.item(row, 2)
            timeout_item = self.members_list.item(row, 3)
            if name is None:
                continue          # row still being built
            try:
                timeout = int(((timeout_item.text() if timeout_item else "") or "10").strip())
            except ValueError:
                timeout = 10
            out.append(RouterMember(
                profile=name.text(),
                model_id=((mid.text() if mid else "") or "").strip(),
                load_on_startup=bool(load is not None and load.checkState() == Qt.Checked),
                stop_timeout=timeout,
            ))
        return out

    def member_pairs(self) -> list:
        """(RouterMember, member Profile) pairs for members whose profile exists."""
        return resolve_member_pairs(self.members(), self.window.router_base_dir())

    def missing_member_profiles(self) -> list:
        """Member profile names that no longer exist on disk.

        Dropping these silently launched a router serving fewer models than the
        list showed, and any harness pinned to the missing id got a 404 with
        nothing in the GUI explaining why."""
        by_name = {p.name: p for p in list_profiles(self.window.router_base_dir())}
        return [m.profile for m in self.members() if m.profile not in by_name]

    def router_issues(self) -> list:
        """Validation issues for the current profile, router context included."""
        p = self.current_profile()
        key_present = bool(api_key_store.resolve_api_key(self.window.router_base_dir(), p)) \
            if p.mode == "router" else False
        issues = validate(p, binary_found=runtime.binary_available(p.runtime.binary),
                          members=self.member_pairs(), api_key_present=key_present,
                          native_binary_ok=native.native_binary_ok_for(p))
        for name in self.missing_member_profiles():
            issues.append(Issue(
                "error",
                f"Member profile {name!r} no longer exists; it would be dropped from "
                f"the router silently. Remove it or recreate the profile."))
        return issues

    def _field_with_browse(self, line_edit: QLineEdit, dot: QWidget | None = None) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(line_edit, 1)
        if dot is not None:
            row.addWidget(dot)
        btn = QPushButton("Browse…")
        btn.clicked.connect(lambda: self._browse_into(line_edit))
        row.addWidget(btn)
        return container

    def _native_binary_field_with_browse(self) -> QWidget:
        """Like `_field_with_browse`, but for `native_binary_edit`.

        `_field_with_browse`'s Browse maps the picked file through the mount
        table (host -> container path) -- correct for a path that will live
        INSIDE a container, wrong here: a native binary is a plain host-side
        executable with no relation to any mount, so mapping it would show
        the "not in a mounted folder" warning (or silently reject the pick)
        for the common case of an empty/unrelated mount table.
        """
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.native_binary_edit, 1)
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._browse_native_binary)
        row.addWidget(btn)
        return container

    def _browse_native_binary(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select llama-server binary", os.path.expanduser("~"))
        if path:
            self.native_binary_edit.setText(path)

    def _browse_into(self, line_edit: QLineEdit) -> None:
        start = os.path.expanduser("~")
        for m in self.mounts_panel.mounts():
            if m.host:
                start = m.host
                break
        path, _ = QFileDialog.getOpenFileName(self, "Select file", start)
        if not path:
            return
        container_path = host_to_container(path, self.mounts_panel.mounts())
        if container_path is None:
            QMessageBox.warning(
                self, "File not in a mounted folder",
                "The selected file is not inside any mounted folder.\n"
                "Add a folder mount that contains it first, then pick it again.")
            return
        line_edit.setText(container_path)

    def load_profile(self, p: Profile) -> None:
        self.window._monitor._stop_log_follower()
        self._profile = p
        self.name_edit.setText(p.name)
        self.image_edit.setText(p.image)
        self.model_edit.setText(p.model)
        self.binary_combo.setCurrentText(p.runtime.binary)
        self.gpu_combo.setCurrentIndex(max(0, self.gpu_combo.findData(p.runtime.gpu_mode)))
        self.engine_combo.blockSignals(True)
        _eidx = self.engine_combo.findData(p.runtime.engine)
        self.engine_combo.setCurrentIndex(_eidx if _eidx >= 0 else 0)
        self.engine_combo.blockSignals(False)
        self._apply_engine_enums()   # extend enums before values are set (no image seeding on load)
        self.mounts_panel.set_mounts(p.mounts)
        self.mmproj_edit.setText(p.mmproj or "")
        self.draft_model_edit.setText(p.draft_model or "")
        self.lora_panel.set_loras(p.loras)
        self.raw_edit.setText(p.raw_args)
        self.extra_args_edit.setText(p.runtime.extra_run_args)
        self.selinux_check.setChecked(p.runtime.selinux_label_disable)
        self.detached_check.setChecked(p.runtime.detached)
        for key, w in self._widgets.items():
            w.set_value(w.setting.default)
            if key in p.settings:
                w.set_value(p.settings[key])
        index = self.mode_combo.findData(p.mode)
        self.mode_combo.setCurrentIndex(index if index >= 0 else 0)
        self.bind_host_combo.setCurrentText(p.runtime.bind_host)
        self.stop_timeout_spin.setValue(p.runtime.stop_timeout)
        idx = self.launch_mode_combo.findData(p.runtime.launch_mode)
        self.launch_mode_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.native_binary_edit.setText(p.runtime.native_binary)
        self._on_launch_mode_changed()
        self.window._monitor._router_statuses = {}
        self.window._monitor._spec_prev = None
        self.window._monitor._props = None
        self.window._monitor._props_model = None
        self.members_list.setRowCount(0)
        for member in p.members:
            self._add_member_item(member)
        self._on_mode_changed()
        self.apply_model_caps()
        self.refresh_preview()
        # Reattach-after-restart is the common path: selecting a saved router
        # must show its key, harness block and exposure banner immediately.
        self.api_key_box.set_scope(p.runtime.router_key_mode)
        self.window.refresh_router_panel_header()

    def current_profile(self) -> Profile:
        # Filtering lives here, in the UI, per the project's catalog contract --
        # never in command_builder. A value left over from the other mode (or
        # loaded from older profile JSON) must not reach argv.
        active = self.active_catalog()
        settings = {}
        for key, w in self._widgets.items():
            if key in active and w.is_set():
                settings[key] = w.value()
        # port is always stored
        settings["port"] = self._widgets["port"].value()
        return Profile(
            name=self._profile_name(),
            image=self.image_edit.text(),
            runtime=Runtime(binary=self.binary_combo.currentText(),
                            gpu_mode=self.gpu_combo.currentData(),
                            selinux_label_disable=self.selinux_check.isChecked(),
                            extra_run_args=self.extra_args_edit.text(),
                            bind_host=self.bind_host_combo.currentText().strip()
                                      or "127.0.0.1",
                            detached=self.detached_check.isChecked(),
                            router_key_mode=self.api_key_box._current_scope(),
                            engine=self.engine_combo.currentData() or "llama.cpp",
                            stop_timeout=self.stop_timeout_spin.value(),
                            launch_mode=self.launch_mode_combo.currentData() or "container",
                            native_binary=self.native_binary_edit.text().strip()),
            mounts=self.mounts_panel.mounts(),
            model=self.model_edit.text(),
            mmproj=self.mmproj_edit.text() or None,
            draft_model=self.draft_model_edit.text() or None,
            loras=self.lora_panel.loras(),
            settings=settings,
            raw_args=self.raw_edit.text(),
            mode=self.mode_combo.currentData() or "server",
            members=self.members(),
        )

    def build_current_command(self, p: Profile | None = None) -> list:
        """argv for the current profile, including the router mount.

        Every path that renders a command (preview, Export .sh, the diagnostic
        report, launch) must go through here: build_command() omits the
        -v <dir>:/router:ro mount unless it is told the host directory, and a
        router command without that mount cannot start -- its --models-preset
        and --api-key-file paths would not exist inside the container.
        """
        p = p or self.current_profile()
        if p.mode != "router":
            return build_command(p, detach=p.runtime.detached)
        return build_command(
            p, router_host_dir=str(api_key_store.router_dir(self.window.router_base_dir(), p.name)))

    def preview_text(self) -> str:
        return " ".join(self.build_current_command())

    def refresh_preview(self) -> None:
        self.preview.setPlainText(self.preview_text())

    def apply_model_caps(self) -> None:
        p = self.current_profile()
        meta, size, caps = (None, None, None)
        if p.model:
            meta, size, caps = model_info.inspect_model(p.model, self.mounts_panel.mounts())
        self._last_caps = caps
        self.model_meta_label.setText(self._meta_caps_text(meta, size, caps))

        described = describe_relevance(caps) if caps else {}
        sugg_by_key, reason_by_key = self._suggestion_index(caps)   # concrete-value suggestions

        for key, widget in self._widgets.items():
            self._apply_dot(widget, key, described, sugg_by_key, reason_by_key)
        self._apply_field_dot(self._mmproj_dot, "mmproj", described, sugg_by_key, reason_by_key)
        self._apply_field_dot(self._draft_model_dot, "draft_model", described, sugg_by_key, reason_by_key)

    def _suggestion_index(self, caps):
        """key -> (Suggestion, reason). A multi-key suggestion indexes each key."""
        sugg_by_key, reason_by_key = {}, {}
        if not caps:
            return sugg_by_key, reason_by_key
        for sg in compute_suggestions(
                caps, self.current_profile().settings,
                mmproj_set=bool(self.mmproj_edit.text()),
                draft_set=bool(self.draft_model_edit.text())):
            for k in list(sg.settings) + list(sg.fields):
                sugg_by_key[k] = sg
                reason_by_key[k] = sg.text
        return sugg_by_key, reason_by_key

    @staticmethod
    def _dot_state_for(key, described, sugg_by_key, reason_by_key):
        tier, reason = described.get(key, (Tier.NEUTRAL, ""))
        state = {"recommended": "suggested", "tune": "suggested",
                 "na": "muted"}.get(getattr(tier, "value", ""), "none")
        return state, reason

    def _apply_dot(self, widget, key, described, sugg_by_key, reason_by_key) -> None:
        state, reason = self._dot_state_for(key, described, sugg_by_key, reason_by_key)
        on_apply = None
        if key in sugg_by_key:
            state = "suggested"
            reason = reason_by_key[key]
            sg = sugg_by_key[key]
            on_apply = lambda s=sg: self._apply_suggestion(s)
        widget.set_suggestion(state, reason, on_apply)

    def _apply_field_dot(self, dot, key, described, sugg_by_key, reason_by_key) -> None:
        state, reason = self._dot_state_for(key, described, sugg_by_key, reason_by_key)
        on_apply = None
        if key in sugg_by_key:
            state = "suggested"
            reason = reason_by_key[key]
            sg = sugg_by_key[key]
            on_apply = lambda s=sg: self._apply_suggestion(s)
        dot.set_state(state, reason, on_apply)

    def _resolve_sibling(self, filename) -> str | None:
        model = self.model_edit.text()
        mounts = self.mounts_panel.mounts()
        for d in (posixpath.dirname(model), posixpath.dirname(posixpath.dirname(model))):
            container = posixpath.join(d, filename)
            host = container_to_host(container, mounts)
            if host and os.path.exists(host):
                return container
        return None

    def _apply_suggestion(self, sg) -> None:
        for key, val in sg.settings.items():
            if key in self._widgets:
                self._widgets[key].set_value(val)
        for field, filename in sg.fields.items():
            container = self._resolve_sibling(filename)
            if container is None:
                continue
            if field == "mmproj":
                self.mmproj_edit.setText(container)
            elif field == "draft_model":
                self.draft_model_edit.setText(container)
        self.refresh_preview()
        self.apply_model_caps()

    @staticmethod
    def _meta_caps_text(meta, size, caps) -> str:
        bits = []
        if size:
            bits.append(f"{size / 1024**3:.1f} GiB")
        if meta and meta.quant:
            bits.append(meta.quant)
        if meta and meta.size_label:
            bits.append(meta.size_label)
        if caps:
            if caps.is_moe:
                bits.append(f"MoE {caps.expert_count}")
            if caps.has_mtp:
                bits.append("MTP" + (" (file)" if caps.mtp_sibling else ""))
            if caps.has_vision:
                bits.append("vision")
            if caps.has_swa:
                bits.append("SWA")
            if caps.ctx_train:
                bits.append(f"{caps.ctx_train // 1024}K ctx")
        return "  ·  ".join(bits)
