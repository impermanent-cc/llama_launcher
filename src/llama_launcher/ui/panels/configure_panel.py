from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QCheckBox,
    QGroupBox, QScrollArea, QLabel, QPlainTextEdit, QPushButton,
)

from llama_launcher.core.settings_catalog import CATALOG
from llama_launcher.ui.widgets.setting_widgets import make_widget, SuggestionDot
from llama_launcher.ui.widgets.no_wheel import NoWheelComboBox
from llama_launcher.ui.panels.mounts_panel import MountsPanel
from llama_launcher.ui.panels.lora_panel import LoraPanel
from llama_launcher.ui.widgets.collapsible import CollapsibleSection
from llama_launcher.ui.widgets.api_key_box import ApiKeyBox
from llama_launcher.ui.widgets.harness_info_box import HarnessInfoBox
from llama_launcher.ui.widgets.status_banner import StatusBanner


class ConfigurePanel(QWidget):
    """The Configure tab body: environment fields, members, settings form,
    command preview + api-key box. Owns all Configure-tab widgets. Holds a
    back-reference to the MainWindow for the few cross-panel calls it makes.
    """

    def __init__(self, window, *, parent=None):
        super().__init__(parent)
        self.window = window
        self.configure_tab = self

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
        self.engine_combo.currentIndexChanged.connect(self.window._on_engine_changed)
        self.gpu_combo = NoWheelComboBox()
        for _gpu_label, _gpu_val in (
            ("CDI — --device nvidia.com/gpu=all (recommended)", "cdi"),
            ("Legacy — --gpus all", "gpus-all"),
            ("None — no GPU passthrough", "none"),
        ):
            self.gpu_combo.addItem(_gpu_label, _gpu_val)
        for w in (self.image_edit, self.model_edit):
            w.textChanged.connect(self.window.refresh_preview)
        self.binary_combo.currentTextChanged.connect(self.window.refresh_preview)
        self.gpu_combo.currentTextChanged.connect(self.window.refresh_preview)
        self.update_badge = QPushButton("")
        self.update_badge.setFlat(True)
        self.update_badge.setVisible(False)
        self.update_badge.clicked.connect(self.window.on_fetch_latest)
        self.detect_image_btn = QPushButton("Detect")
        self.detect_image_btn.setToolTip(
            "Fill the Image field from llama.cpp images already pulled locally "
            "(podman/docker images).")
        self.detect_image_btn.clicked.connect(self.window.detect_image)
        self.fetch_btn = QPushButton("Fetch latest")
        self.fetch_btn.setToolTip(
            "Query the container registry (GHCR) for the newest build tag matching "
            "this image's repo and variant, and update the Image field. Requires an "
            "image to be set (use Detect or type one). This updates the tag only — it "
            "does NOT download the build; pull it with podman/docker pull.")
        self.fetch_btn.clicked.connect(self.window.on_fetch_latest)
        image_row = QHBoxLayout()
        image_row.setContentsMargins(0, 0, 0, 0)
        image_row.addWidget(self.image_edit, 1)
        image_row.addWidget(self.detect_image_btn)
        image_row.addWidget(self.fetch_btn)
        image_row.addWidget(self.update_badge)
        image_widget = QWidget()
        image_widget.setLayout(image_row)
        from PySide6.QtWidgets import QTableWidget, QHeaderView

        # NoWheel: an unfocused scroll over these must not silently flip launch
        # semantics (terminal vs detached) or expose the port to the network.
        self.mode_combo = NoWheelComboBox()
        self.mode_combo.addItem("Single server", "server")
        self.mode_combo.addItem("Router (headless host)", "router")
        self.mode_combo.currentIndexChanged.connect(self.window._on_mode_changed)

        self.bind_host_combo = NoWheelComboBox()
        self.bind_host_combo.setEditable(True)
        self.bind_host_combo.addItems(["127.0.0.1", "0.0.0.0"])
        self.bind_host_combo.currentTextChanged.connect(self.window.refresh_preview)

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
        self.members_list.itemChanged.connect(lambda _i: self.window.refresh_preview())
        self.add_member_btn = QPushButton("Add member…")
        self.add_member_btn.clicked.connect(self.window._on_add_member)
        self.remove_member_btn = QPushButton("Remove member")
        self.remove_member_btn.clicked.connect(self.window._on_remove_member)
        self.edit_member_btn = QPushButton("Edit member…")
        self.edit_member_btn.setToolTip(
            "Load the selected member's profile into the form to set its GPU "
            "layers, MoE offload, context, etc. (double-clicking a row does the same).")
        self.edit_member_btn.clicked.connect(self.window._on_edit_member)
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
            lambda _r, _c: self.window._on_edit_member() if _c == 0 else None)

        left_form.addRow("Mode", self.mode_combo)
        left_form.addRow("Bind address", self.bind_host_combo)
        left_form.addRow("Router members", members_widget)

        left_form.addRow("Engine", self.engine_combo)
        left_form.addRow("Image", image_widget)
        left_form.addRow("Model", self.window._field_with_browse(self.model_edit))
        left_form.addRow("Runtime", self.binary_combo)
        left_form.addRow("GPU", self.gpu_combo)

        # mounts editor
        self.mounts_panel = MountsPanel()
        self.mounts_panel.changed.connect(self.window.refresh_preview)
        left_form.addRow("Folders", self.mounts_panel)
        self.mmproj_edit = QLineEdit(); self.mmproj_edit.textChanged.connect(self.window.refresh_preview)
        self.draft_model_edit = QLineEdit(); self.draft_model_edit.textChanged.connect(self.window.refresh_preview)
        self.raw_edit = QLineEdit(); self.raw_edit.textChanged.connect(self.window.refresh_preview)
        self._mmproj_dot = SuggestionDot(self)
        self._draft_model_dot = SuggestionDot(self)
        left_form.addRow("mmproj", self.window._field_with_browse(self.mmproj_edit, self._mmproj_dot))
        left_form.addRow("draft model", self.window._field_with_browse(self.draft_model_edit, self._draft_model_dot))
        self.lora_panel = LoraPanel()
        self.lora_panel.changed.connect(self.window.refresh_preview)
        self.lora_section = CollapsibleSection("LoRA adapters", self.lora_panel, collapsed=True)
        left_form.addRow(self.lora_section)
        left_form.addRow("Raw args", self.raw_edit)
        self.extra_args_edit = QLineEdit()
        self.extra_args_edit.textChanged.connect(self.window.refresh_preview)
        left_form.addRow("Extra podman args", self.extra_args_edit)
        self.selinux_check = QCheckBox("Disable SELinux labels (--security-opt=label=disable)")
        self.selinux_check.toggled.connect(self.window.refresh_preview)
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
            w.changed.connect(self.window.refresh_preview)
            self._widgets[key] = w
            groups[setting.group].addRow(setting.flag, w)
            self._setting_rows[key] = (groups[setting.group], w)
        # When load-mode is active it supersedes --no-mmap/--mlock (see
        # command_builder), so gray those legacy checkboxes out to signal they're
        # ignored. Kept in sync on edit, mode change and profile load.
        if "load-mode" in self._widgets:
            self._widgets["load-mode"].changed.connect(self.window._sync_load_mode_legacy)
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
        self.model_edit.textChanged.connect(lambda _: self.window.apply_model_caps())
        self.mounts_panel.changed.connect(self.window.apply_model_caps)
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
        self.export_sh_btn.clicked.connect(self.window._on_export_sh)
        config_bottom_box.addWidget(QLabel("Command preview:"))
        config_bottom_box.addLayout(preview_row)

        # lifecycle / launch buttons (added to a shared row in MainWindow, below
        # the tab widget, so they're visible regardless of the active tab)
        self.launch_btn = QPushButton("▶ Launch")
        self.stop_btn = QPushButton("■ Stop")
        self.restart_btn = QPushButton("⟳ Restart")
        self.web_ui_btn = QPushButton("Open Web UI")
        self.web_ui_btn.setEnabled(False)
        self.web_ui_btn.clicked.connect(self.window.open_web_ui)
        self.detached_check = QCheckBox("Run detached (no terminal window)")
        self.detached_check.setToolTip(
            "Launch without a terminal window; watch output on the Monitor "
            "tab and use Stop to shut it down.")
        self.detached_check.toggled.connect(self.window.refresh_preview)

        # profile bar widgets (assembled into the top bar, alongside
        # MainWindow-owned widgets, in MainWindow.__init__)
        self.name_edit = QLineEdit(self.window._profile.name)
        self.name_edit.setPlaceholderText("Profile name")
        self.name_edit.setToolTip(
            "Name for this profile. Also sets the container name "
            "(llama-<name>), so pick something filesystem-friendly.")
        self.name_edit.textChanged.connect(self.window.refresh_preview)
        self.profile_combo = NoWheelComboBox()
        self.save_btn = QPushButton("Save")
        self.save_as_btn = QPushButton("Save As")
        self.delete_btn = QPushButton("Delete")
        self.report_btn = QPushButton("Generate report")
        self.save_btn.clicked.connect(self.window.save_current_profile)
        self.save_as_btn.clicked.connect(self.window.save_as_profile)
        self.delete_btn.clicked.connect(self.window.delete_current_profile)
        self.report_btn.clicked.connect(self.window.on_generate_report)
        self.profile_combo.activated.connect(self.window._on_pick_profile)

        # lifecycle buttons
        self.launch_btn.clicked.connect(self.window.on_launch)
        self.stop_btn.clicked.connect(self.window.on_stop)
        self.restart_btn.clicked.connect(self.window.on_restart)
