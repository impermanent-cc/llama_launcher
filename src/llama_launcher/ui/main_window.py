import dataclasses
import os
import subprocess
from pathlib import Path

import datetime

from PySide6.QtCore import Qt, QObject, QThread, Signal
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit,
    QCheckBox, QGroupBox, QScrollArea, QLabel, QPlainTextEdit, QPushButton,
    QMessageBox, QFileDialog, QInputDialog, QTabWidget
)

from llama_launcher.core.spec import (
    Profile, Mount, Runtime, RouterMember, member_model_id, slugify,
)
from llama_launcher.core.router_preset import render_preset
from llama_launcher.core.settings_catalog import CATALOG, member_catalog, router_catalog
from llama_launcher.core.command_builder import build_command
from llama_launcher.core.pathmap import host_to_container
from llama_launcher.core.validation import (
    validate, Issue, LOOPBACK_HOSTS, dial_host,
)
from llama_launcher.store.profiles import (
    default_base_dir, list_profiles, save_profile, delete_profile,
    load_config, save_config, profile_to_dict, resolve_member_pairs,
)
from llama_launcher.core.instances import Instance, build_instances
from llama_launcher.core.presets import PRESETS, Preset, preset_suggestions
from llama_launcher.store.presets import list_presets as list_user_presets, save_preset as save_user_preset
from llama_launcher.services import runtime, terminal, registry, health, metrics, gpu, model_info
from llama_launcher.services import benchmark, benchmark_store
from llama_launcher.core import vram
from llama_launcher.core.mtp_stats import spec_counters, spec_delta
from llama_launcher.core import report as report_mod
from llama_launcher.services.registry import split_image, variant_prefix
from llama_launcher.ui.dialogs.report_dialog import ReportDialog
import posixpath
from llama_launcher.core.capabilities import relevance, Tier, suggestions as compute_suggestions
from llama_launcher.core.pathmap import container_to_host
from llama_launcher.ui.widgets.setting_widgets import make_widget, TIER_QSS
from llama_launcher.ui.widgets.no_wheel import NoWheelComboBox
from llama_launcher.ui.panels.mounts_panel import MountsPanel
from llama_launcher.ui.panels.lora_panel import LoraPanel
from llama_launcher.ui.widgets.collapsible import CollapsibleSection
from llama_launcher.ui.panels.monitor_panel import MonitorPanel
from llama_launcher.ui.panels.router_panel import RouterPanel
from llama_launcher.services import api_key as api_key_store
from llama_launcher.services import router_api


def _fmt_uptime(started_at: str | None) -> str:
    """Return a short human-readable uptime string from a started_at ISO timestamp."""
    if not started_at:
        return ""
    try:
        dt = datetime.datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        elapsed = int((now - dt).total_seconds())
        if elapsed < 60:
            return f"{elapsed}s"
        if elapsed < 3600:
            return f"{elapsed // 60}m"
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        return f"{h}h{m:02d}m"
    except (ValueError, TypeError):
        return ""


class _UpdateWorker(QThread):
    found = Signal(str)

    def __init__(self, repo: str, prefix: str, parent=None):
        super().__init__(parent)
        self._repo = repo
        self._prefix = prefix

    def run(self):
        try:
            tag = registry.fetch_latest(self._repo, self._prefix)
            if tag:
                self.found.emit(tag)
        except Exception:
            pass


class BenchmarkWorker(QObject):
    """Runs benchmark.run_benchmark() off the UI thread.

    Built with an already-constructed client/snapshot/timestamp (endpoint
    derivation and profile reads happen on the UI thread, before this worker
    is started) so run() touches no GUI/profile state -- it only calls
    run_benchmark() and emits a result signal. Qt delivers finished/failed to
    the UI-thread slot via a queued connection since this object lives on a
    different thread than MainWindow.
    """
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, client, sizes, n_predict, warmup, repeats, snapshot,
                 timestamp, parent=None):
        super().__init__(parent)
        self._client = client
        self._sizes = sizes
        self._n_predict = n_predict
        self._warmup = warmup
        self._repeats = repeats
        self._snapshot = snapshot
        self._timestamp = timestamp
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _should_cancel(self) -> bool:
        return self._cancelled

    def run(self) -> None:
        try:
            run = benchmark.run_benchmark(
                self._client, self._sizes, self._n_predict, self._warmup,
                self._repeats, self._snapshot, self._timestamp,
                should_cancel=self._should_cancel)
        except benchmark.BenchmarkError as e:
            self.failed.emit(str(e))
            return
        self.finished.emit(run)


def base_dir():
    return default_base_dir()


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Llama Launcher")
        self._profile = Profile(name="New Profile")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Configure tab = the existing left+right body
        configure_tab = QWidget()
        body = QHBoxLayout(configure_tab)

        # LEFT: environment (image + model only for v1 binding; mounts editor TODO-UI)
        left = QGroupBox("Environment")
        left_form = QFormLayout(left)
        self.image_edit = QLineEdit()
        self.model_edit = QLineEdit()
        self.binary_combo = NoWheelComboBox(); self.binary_combo.addItems(["podman", "docker"])
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
        self.update_badge.clicked.connect(self.on_fetch_latest)
        self.detect_image_btn = QPushButton("Detect")
        self.detect_image_btn.setToolTip(
            "Fill the Image field from llama.cpp images already pulled locally "
            "(podman/docker images).")
        self.detect_image_btn.clicked.connect(self.detect_image)
        image_row = QHBoxLayout()
        image_row.setContentsMargins(0, 0, 0, 0)
        image_row.addWidget(self.image_edit, 1)
        image_row.addWidget(self.detect_image_btn)
        image_row.addWidget(self.update_badge)
        image_widget = QWidget()
        image_widget.setLayout(image_row)
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

        # A table, not a list: model id / load-on-startup / stop-timeout are all
        # per-member settings the spec requires to be editable, and inline
        # editing keeps them reachable without a modal dialog (which would hang
        # the headless test run).
        self.members_list = QTableWidget(0, 4)
        self.members_list.setHorizontalHeaderLabels(
            ["Profile", "Model id (harness)", "Load at start", "Stop timeout (s)"])
        self.members_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.members_list.verticalHeader().setVisible(False)
        self.members_list.setMaximumHeight(140)
        self.members_list.itemChanged.connect(lambda _i: self.refresh_preview())
        self.add_member_btn = QPushButton("Add member…")
        self.add_member_btn.clicked.connect(self._on_add_member)
        self.remove_member_btn = QPushButton("Remove member")
        self.remove_member_btn.clicked.connect(self._on_remove_member)
        members_row = QHBoxLayout()
        members_row.setContentsMargins(0, 0, 0, 0)
        members_row.addWidget(self.add_member_btn)
        members_row.addWidget(self.remove_member_btn)
        members_widget = QWidget()
        members_box = QVBoxLayout(members_widget)
        members_box.setContentsMargins(0, 0, 0, 0)
        members_box.addWidget(self.members_list)
        members_box.addLayout(members_row)

        left_form.addRow("Mode", self.mode_combo)
        left_form.addRow("Bind address", self.bind_host_combo)
        left_form.addRow("Router members", members_widget)

        left_form.addRow("Image", image_widget)
        left_form.addRow("Model", self._field_with_browse(self.model_edit))
        left_form.addRow("Runtime", self.binary_combo)
        left_form.addRow("GPU", self.gpu_combo)

        # mounts editor
        self.mounts_panel = MountsPanel()
        self.mounts_panel.changed.connect(self.refresh_preview)
        left_form.addRow("Folders", self.mounts_panel)
        self.mmproj_edit = QLineEdit(); self.mmproj_edit.textChanged.connect(self.refresh_preview)
        self.draft_model_edit = QLineEdit(); self.draft_model_edit.textChanged.connect(self.refresh_preview)
        self.raw_edit = QLineEdit(); self.raw_edit.textChanged.connect(self.refresh_preview)
        left_form.addRow("mmproj", self._field_with_browse(self.mmproj_edit))
        left_form.addRow("draft model", self._field_with_browse(self.draft_model_edit))
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
        self.fetch_btn = QPushButton("Fetch latest build")
        self.fetch_btn.clicked.connect(self.on_fetch_latest)
        left_form.addRow(self.fetch_btn)

        body.addWidget(left, 3)

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
        right_scroll.setWidget(right_inner)
        body.addWidget(right_scroll, 2)
        self.setStyleSheet((self.styleSheet() or "") + TIER_QSS)
        self._last_caps = None
        self._preset_family = None
        self._router_statuses: dict = {}
        self._spec_prev = None      # previous /metrics spec-decode counter read
        self._props = None          # cached /props for the current model load
        self._props_model = None    # router-polled model id the cache is keyed on
        self._benchmark_thread = None
        self._benchmark_worker = None
        self._benchmark_profile_name = None
        self.configure_tab = configure_tab

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(configure_tab, "Configure")
        self.monitor_panel = MonitorPanel()
        self.monitor_panel.enable_metrics_requested.connect(self._on_enable_metrics)
        self.monitor_panel.benchmark_run_requested.connect(self._on_benchmark_run)
        self.monitor_panel.benchmark_cancel_requested.connect(self._on_benchmark_cancel)
        self.monitor_panel.instance_selected.connect(self._on_instance_selected)
        self.monitor_panel.instance_stop_requested.connect(self._on_instance_stop)
        self.tabs.addTab(self.monitor_panel, "Monitor")
        self.router_panel = RouterPanel()
        self.router_panel.load_requested.connect(self._on_router_load)
        self.router_panel.unload_requested.connect(self._on_router_unload)
        self.router_panel.regenerate_requested.connect(self._on_regenerate_key)
        self.tabs.addTab(self.router_panel, "Router")
        root.addWidget(self.tabs)

        # BOTTOM: preview + buttons (shared, below both tabs)
        self.model_meta_label = QLabel("")
        root.addWidget(self.model_meta_label)
        from PySide6.QtWidgets import QHBoxLayout as _HBox
        self.suggestions_strip = QWidget()
        self._suggestions_layout = _HBox(self.suggestions_strip)
        self._suggestions_layout.setContentsMargins(0, 0, 0, 0)
        family_row = QHBoxLayout()
        family_row.addWidget(QLabel("Suggest for family"))
        self.family_combo = NoWheelComboBox()
        self.family_combo.activated.connect(self._on_pick_family)
        family_row.addWidget(self.family_combo, 1)
        self.save_preset_btn = QPushButton("Save as preset…")
        self.save_preset_btn.clicked.connect(self.on_save_preset)
        family_row.addWidget(self.save_preset_btn)
        root.addLayout(family_row)
        self._reload_family_combo()
        root.addWidget(self.suggestions_strip)
        self.model_edit.textChanged.connect(lambda _: self.apply_model_caps())
        self.mounts_panel.changed.connect(self.apply_model_caps)
        preview_row = QHBoxLayout()
        self.preview = QPlainTextEdit(); self.preview.setReadOnly(True)
        preview_row.addWidget(self.preview, 1)
        self.export_sh_btn = QPushButton("Export .sh")
        self.export_sh_btn.clicked.connect(self._on_export_sh)
        preview_row.addWidget(self.export_sh_btn)
        root.addWidget(QLabel("Command preview:"))
        root.addLayout(preview_row)
        buttons = QHBoxLayout()
        self.launch_btn = QPushButton("▶ Launch")
        self.stop_btn = QPushButton("■ Stop")
        self.restart_btn = QPushButton("⟳ Restart")
        self.web_ui_btn = QPushButton("Open Web UI")
        self.web_ui_btn.setEnabled(False)
        self.web_ui_btn.clicked.connect(self.open_web_ui)
        for b in (self.launch_btn, self.stop_btn, self.restart_btn, self.web_ui_btn):
            buttons.addWidget(b)
        self.detached_check = QCheckBox("Run detached (no terminal window)")
        self.detached_check.setToolTip(
            "Launch without a terminal window; watch output on the Monitor "
            "tab and use Stop to shut it down.")
        self.detached_check.toggled.connect(self.refresh_preview)
        buttons.addWidget(self.detached_check)
        root.addLayout(buttons)

        # profile bar (added to the top of root via insertLayout)
        bar = QHBoxLayout()
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
        self.status_label = QLabel("● stopped")
        bar.addWidget(QLabel("Name"))
        bar.addWidget(self.name_edit, 1)
        bar.addWidget(self.profile_combo, 1)
        for b in (self.save_btn, self.save_as_btn, self.delete_btn, self.report_btn):
            bar.addWidget(b)
        bar.addWidget(self.status_label)
        root.insertLayout(0, bar)
        self.save_btn.clicked.connect(self.save_current_profile)
        self.save_as_btn.clicked.connect(self.save_as_profile)
        self.delete_btn.clicked.connect(self.delete_current_profile)
        self.report_btn.clicked.connect(self.on_generate_report)
        self.profile_combo.activated.connect(self._on_pick_profile)

        # lifecycle buttons
        self.launch_btn.clicked.connect(self.on_launch)
        self.stop_btn.clicked.connect(self.on_stop)
        self.restart_btn.clicked.connect(self.on_restart)

        self._reload_profile_list()

        self.refresh_preview()

        self._log_proc = None
        self._stop_proc = None
        self._active_instance = None      # Instance being monitored, or None -> current profile
        self._instances = []              # last-built Instance list (for selection lookup)

        from PySide6.QtWidgets import QSystemTrayIcon, QMenu, QStyle
        from llama_launcher.ui.icon import app_icon
        self._really_quit = False
        # Apply our own SVG icon to the window. app.main() also sets it at the
        # application level, but doing it here means the icon is present even
        # when the window is constructed without going through main() (tests,
        # embedding), and is what the window-manager titlebar/Alt-Tab uses.
        self.setWindowIcon(app_icon())
        # Minimize-to-tray is OPT-IN: enabled only when the user turns it on in
        # config AND the desktop actually provides a system tray. By default,
        # closing the window quits the app (see closeEvent), so a tray-less or
        # half-wired session never strands the process behind a blocked terminal.
        self._minimize_to_tray = (
            bool(load_config(base_dir()).get("minimize_to_tray", False))
            and QSystemTrayIcon.isSystemTrayAvailable()
        )
        if self._minimize_to_tray:
            self.tray = QSystemTrayIcon(self)
            tray_icon = app_icon()
            if tray_icon.isNull():   # no asset/theme icon found — keep a visible fallback
                tray_icon = self.style().standardIcon(QStyle.SP_ComputerIcon)
            self.tray.setIcon(tray_icon)
            self.tray.setToolTip("Llama Launcher")
            menu = QMenu()
            menu.addAction("Show", self.showNormal)
            menu.addAction("Launch", self.on_launch)
            menu.addAction("Stop", self.on_stop)
            menu.addSeparator()
            menu.addAction("Quit", self.quit_app)
            self.tray.setContextMenu(menu)
            self.tray.show()
        else:
            self.tray = None

        self.lora_panel.set_browse_resolver(
            lambda h: host_to_container(h, self.mounts_panel.mounts())
        )

        # Auto-insert the local image when there's exactly one and none is set yet.
        self._autofill_image_if_empty()

        from PySide6.QtCore import QTimer
        self._status_timer = QTimer(self)
        interval = load_config(base_dir()).get("monitor_interval_ms", 2000)
        self._status_timer.setInterval(interval)
        self._status_timer.timeout.connect(self.update_status)
        self._status_timer.start()

        if load_config(base_dir()).get("update_check", True):
            self._update_timer = QTimer(self)
            self._update_timer.setSingleShot(True)
            self._update_timer.timeout.connect(self.run_update_check)
            self._update_timer.start(3000)

    def _profile_name(self) -> str:
        """Current profile name from the Name field, falling back to a default
        so the container name / preview are never empty."""
        return self.name_edit.text().strip() or "New Profile"

    def _container_name(self) -> str:
        return f"llama-{slugify(self._profile_name())}"

    def active_catalog(self) -> dict:
        """The settings that apply to the mode currently selected in the form.

        Router CLI args outrank every member's preset value, so a router must
        not be able to carry model-level settings; conversely --models-max on a
        single-model llama-server is rejected outright.
        """
        return (router_catalog() if self.mode_combo.currentData() == "router"
                else member_catalog())

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
        for w in (self.members_list, self.add_member_btn, self.remove_member_btn):
            w.setVisible(is_router)
        self._apply_mode_to_settings_form()
        # A router has no model of its own; its members carry those fields.
        for w in (self.model_edit, self.mmproj_edit, self.draft_model_edit):
            w.setEnabled(not is_router)
        self.lora_panel.setEnabled(not is_router)
        self.detached_check.setVisible(not is_router)
        self.refresh_preview()

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

    def _on_add_member(self) -> None:
        names = [p.name for p in list_profiles(base_dir())
                 if p.mode != "router" and p.name != self._profile_name()]
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
        return resolve_member_pairs(self.members(), base_dir())

    def missing_member_profiles(self) -> list:
        """Member profile names that no longer exist on disk.

        Dropping these silently launched a router serving fewer models than the
        list showed, and any harness pinned to the missing id got a 404 with
        nothing in the GUI explaining why."""
        by_name = {p.name: p for p in list_profiles(base_dir())}
        return [m.profile for m in self.members() if m.profile not in by_name]

    def router_issues(self) -> list:
        """Validation issues for the current profile, router context included."""
        p = self.current_profile()
        key_present = bool(api_key_store.read_api_key(self.router_base_dir(), p.name)) \
            if p.mode == "router" else False
        issues = validate(p, binary_found=runtime.binary_available(p.runtime.binary),
                          members=self.member_pairs(), api_key_present=key_present)
        for name in self.missing_member_profiles():
            issues.append(Issue(
                "error",
                f"Member profile {name!r} no longer exists; it would be dropped from "
                f"the router silently. Remove it or recreate the profile."))
        return issues

    def _field_with_browse(self, line_edit: QLineEdit) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(line_edit, 1)
        btn = QPushButton("Browse…")
        btn.clicked.connect(lambda: self._browse_into(line_edit))
        row.addWidget(btn)
        return container

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
        self._stop_log_follower()
        self._profile = p
        self.name_edit.setText(p.name)
        self.image_edit.setText(p.image)
        self.model_edit.setText(p.model)
        self.binary_combo.setCurrentText(p.runtime.binary)
        self.gpu_combo.setCurrentIndex(max(0, self.gpu_combo.findData(p.runtime.gpu_mode)))
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
        self._router_statuses = {}
        self._spec_prev = None
        self._props = None
        self._props_model = None
        self.members_list.setRowCount(0)
        for member in p.members:
            self._add_member_item(member)
        self._on_mode_changed()
        self.apply_model_caps()
        self.refresh_preview()
        # Reattach-after-restart is the common path: selecting a saved router
        # must show its key, harness block and exposure banner immediately.
        self.refresh_router_panel_header()

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
                            detached=self.detached_check.isChecked()),
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
            p, router_host_dir=str(api_key_store.router_dir(self.router_base_dir(), p.name)))

    def preview_text(self) -> str:
        return " ".join(self.build_current_command())

    def refresh_preview(self) -> None:
        self.preview.setPlainText(self.preview_text())

    def _reload_profile_list(self):
        self.profile_combo.clear()
        self._profiles = {p.name: p for p in list_profiles(base_dir())}
        self.profile_combo.addItems(list(self._profiles.keys()))

    def _on_pick_profile(self, _index):
        self._stop_log_follower()
        name = self.profile_combo.currentText()
        if name in self._profiles:
            self.load_profile(self._profiles[name])

    def save_current_profile(self):
        p = self.current_profile()       # name comes from the Name field
        self._profile = p
        save_profile(p, base_dir())
        self._reload_profile_list()
        self.profile_combo.setCurrentText(p.name)

    def save_as_profile(self):
        name, ok = QInputDialog.getText(self, "Save As", "Profile name:",
                                        text=self._profile_name())
        if ok and name:
            self.name_edit.setText(name)
            self.save_current_profile()

    def delete_current_profile(self):
        name = self.profile_combo.currentText()
        if name:
            delete_profile(name, base_dir())
            self._reload_profile_list()

    def router_base_dir(self):
        return base_dir()

    def router_api_key(self) -> str:
        return api_key_store.ensure_api_key(self.router_base_dir(), self._profile_name())

    def prepare_router_files(self) -> tuple:
        """Write models.ini + api-key for the current router. Returns (dir, warnings)."""
        name = self._profile_name()
        result = render_preset(self.member_pairs())
        api_key_store.ensure_api_key(self.router_base_dir(), name)
        api_key_store.write_preset(self.router_base_dir(), name, result.text)
        return str(api_key_store.router_dir(self.router_base_dir(), name)), result.warnings

    def _on_regenerate_key(self) -> None:
        name = self._profile_name()
        answer = QMessageBox.question(
            self, "Regenerate API key",
            "This invalidates the key any configured harness is using. Continue?")
        if answer != QMessageBox.Yes:
            return
        api_key_store.regenerate_api_key(self.router_base_dir(), name)
        self.refresh_router_panel_header()

    def refresh_router_panel_header(self) -> None:
        p = self.current_profile()
        if p.mode != "router":
            return
        host = p.runtime.bind_host
        display_host = dial_host(host)
        port = p.settings.get("port", 8080)
        # A router without a key is unusable, and the harness block exists so the
        # key can be copied BEFORE the first launch. Generating here is
        # idempotent and is the only side effect on this path.
        key = api_key_store.ensure_api_key(self.router_base_dir(), p.name)
        self.router_panel.set_endpoint(
            f"http://{display_host}:{port}", key,
            [member_model_id(m) for m in p.members])
        self.router_panel.set_exposure_warning(
            f"Bound to {host}: reachable beyond this machine. The API key is required."
            if host not in LOOPBACK_HOSTS else "")

    def _update_spec_stats(self, p: Profile) -> None:
        """Derive spec-decode acceptance from /metrics counters.

        The log-scraped acceptance line stays the source when --metrics is off,
        and is the only source in server mode. In router mode the log belongs to
        the ROUTER, so a child model's acceptance can only be attributed via
        ?model=<id> -- which is what these counters are for.
        """
        if not p.settings.get("metrics"):
            return
        model_scope = self._router_pollable_model() if p.mode == "router" else None
        if p.mode == "router" and model_scope is None:
            return
        text = metrics.fetch_metrics_text(
            p.settings.get("port", 8080), model=model_scope,
            api_key=api_key_store.read_api_key(self.router_base_dir(), p.name)
            if p.mode == "router" else None,
            host=dial_host(p.runtime.bind_host))
        cur = spec_counters(text) if text else None
        if cur is None:
            return
        if self._spec_prev is not None:
            self.monitor_panel.set_draft_stats(spec_delta(self._spec_prev, cur),
                                               source="counters")
        self._spec_prev = cur

    def _refresh_props(self, p: Profile) -> str | None:
        """Fetch /props once per model load and cache it (static per load).

        Keyed on the router-polled model id so a router swap re-fetches; in
        single-model mode the key is constant, so it fetches exactly once.

        Returns the router model key it resolved (None in server mode, or
        when router mode has nothing loaded), so callers that already need
        that value -- e.g. the benchmark-availability gate -- can reuse it
        instead of polling `_router_pollable_model()` again.
        """
        port = p.settings.get("port", 8080)
        if p.mode == "router":
            host = self._router_host(p)
            key = api_key_store.read_api_key(self.router_base_dir(), p.name)
            model_key = self._router_pollable_model()
        else:
            host = dial_host(p.runtime.bind_host)
            key, model_key = None, None
        if self._props is not None and self._props_model == model_key:
            return model_key
        info = metrics.fetch_props(port, api_key=key, host=host)
        if info is None:
            return model_key           # leave cache empty; retry next ready poll
        self._props = info
        self._props_model = model_key
        self.monitor_panel.set_props(info)
        return model_key

    def _report_launch_error(self, text: str = None, *, show_dialog: bool = False) -> None:
        """Show why a detached launch -- router or server -- failed to start.
        Routed to the router panel (non-modal: this fires from a QProcess
        signal, which tests drive); a detached SERVER launch also pops a
        QMessageBox, since a Monitor-tab-only user may never see the router
        panel and would otherwise miss the failure entirely.

        `show_dialog` is decided by the CALLER at launch time (when the
        profile's mode is known synchronously), not re-derived here from
        live UI state: this fires from an async QProcess callback, possibly
        seconds later, by which point the user may have switched profiles
        or flipped the mode combo -- current_profile() at that moment would
        no longer describe the launch that actually failed."""
        self.status_label.setText("● failed to start")
        reason = (f"launch failed: {text.splitlines()[-1][:200]}"
                  if text else "launch failed")
        self.router_panel.set_error(reason)
        if show_dialog:
            QMessageBox.critical(self, "Launch failed", reason)

    def adopt_running_containers(self) -> list:
        """Containers this launcher owns, so a detached router survives a GUI restart."""
        p = self.current_profile()
        return runtime.list_launcher_containers(p.runtime.binary)

    def _router_host(self, p: Profile) -> str:
        """The address the GUI itself dials for this profile."""
        return dial_host(p.runtime.bind_host)

    def refresh_router_models(self) -> None:
        p = self.current_profile()
        if p.mode != "router":
            return
        host = self._router_host(p)
        port = p.settings.get("port", 8080)
        key = api_key_store.read_api_key(self.router_base_dir(), p.name)
        models = router_api.list_models(host, port, key)
        if models is None:            # unreachable, as opposed to serving nothing
            self._router_statuses = {}
            self.router_panel.set_models([])
            self.router_panel.set_connected(False)
            return
        self._router_statuses = {m.id: m.status for m in models}
        self.router_panel.set_models(models)
        self.router_panel.set_connected(True)

    def _router_pollable_model(self) -> str | None:
        """The resident model to scope Monitor polling to, or None.

        Sleeping and unloaded models are skipped: there is nothing to measure,
        and not polling them is the whole point of an idle-unloading host."""
        for model_id, status in self._router_statuses.items():
            if status == "loaded":
                return model_id
        return None

    def _on_router_load(self, model_id: str) -> None:
        p = self.current_profile()
        key = api_key_store.read_api_key(self.router_base_dir(), p.name)
        ok = router_api.load_model(self._router_host(p), p.settings.get("port", 8080),
                                   key, model_id)
        self.refresh_router_models()
        if not ok:
            # Silently discarding this left a failed load looking identical to a
            # slow one: the row just stayed "unloaded" forever.
            self.router_panel.set_error(f"load failed: {model_id}")

    def _on_router_unload(self, model_id: str) -> None:
        p = self.current_profile()
        key = api_key_store.read_api_key(self.router_base_dir(), p.name)
        ok = router_api.unload_model(self._router_host(p), p.settings.get("port", 8080),
                                     key, model_id)
        self.refresh_router_models()
        if not ok:
            self.router_panel.set_error(f"unload failed: {model_id}")

    def _validate_or_warn(self) -> bool:
        p = self.current_profile()
        # The key must exist before the exposure rule is evaluated: a router
        # always gets one at launch, but this runs before prepare_router_files.
        if p.mode == "router":
            api_key_store.ensure_api_key(self.router_base_dir(), p.name)
        issues = self.router_issues()
        errors = [i for i in issues if i.level == "error"]
        if errors:
            QMessageBox.critical(self, "Cannot launch",
                                 "\n".join(i.message for i in errors))
            return False
        warns = [i for i in issues if i.level == "warning"]
        if warns:
            QMessageBox.warning(self, "Warnings", "\n".join(i.message for i in warns))
        return True

    def vram_check(self) -> str | None:
        p = self.current_profile()
        meta, weights, _caps = model_info.inspect_model(p.model, self.mounts_panel.mounts()) if p.model else (None, None, None)
        free = gpu.free_vram_bytes()
        if meta is None or free is None or not meta.n_layers or not meta.n_embd:
            return None
        ctx = p.settings.get("ctx-size") or meta.ctx_train or 4096
        est = vram.estimate(
            n_layers=meta.n_layers, n_head=meta.n_head or 1,
            n_head_kv=meta.n_head_kv or meta.n_head or 1, n_embd=meta.n_embd, ctx=ctx,
            k_quant=p.settings.get("cache-type-k", "f16"),
            v_quant=p.settings.get("cache-type-v", "f16"),
            weights_bytes=weights or 0,
        )
        ok, margin = vram.fits(est.total_bytes, free)
        if ok:
            return None
        gib = 1024 ** 3
        return (f"Estimated VRAM need ~{est.total_bytes/gib:.1f} GiB exceeds free "
                f"~{free/gib:.1f} GiB by ~{-margin/gib:.1f} GiB. It may not fit — "
                f"consider quantized KV cache (-ctk/-ctv q8_0) or a higher --n-cpu-moe. "
                f"(Estimate is conservative; --n-cpu-moe/-ngl reduce actual GPU use.)")

    def on_launch(self):
        if not self._validate_or_warn():
            return
        p = self.current_profile()

        if p.mode == "router":
            router_host_dir, warnings = self.prepare_router_files()
            if warnings:
                QMessageBox.warning(self, "Preset warnings", "\n".join(warnings))
            argv = build_command(p, router_host_dir=router_host_dir)
            # Relaunching over a LIVE router would drop a resident model and any
            # in-flight harness requests, so confirm before tearing it down.
            if runtime.container_state(self._container_name(),
                                       p.runtime.binary) == "running":
                answer = QMessageBox.question(
                    self, "Router already running",
                    "This router is already running. Relaunching stops it, "
                    "unloading any resident model and dropping in-flight "
                    "requests. Continue?")
                if answer != QMessageBox.Yes:
                    return

            # A stopped container of the same name would block the new run,
            # since router mode deliberately omits --rm. Chain rather than fire
            # both at once: _spawn_async is asynchronous, so an unchained run
            # would race the removal and lose with "name already in use".
            self.monitor_panel.reset()
            self.monitor_panel.set_benchmark_history(
                benchmark_store.load(default_base_dir(), p.name))
            self._spec_prev = None
            self._props = None
            self._props_model = None
            self._spawn_async(
                runtime.rm_argv(self._container_name(), p.runtime.binary),
                on_done=lambda: self._spawn_async(
                    argv, on_done=self.update_status,
                    # Detached means no terminal, so a bad image ref or a CDI
                    # failure would otherwise produce nothing but a status label
                    # stuck on "stopped".
                    on_error=self._report_launch_error))
            self.refresh_router_panel_header()
            return

        warn = self.vram_check()
        if warn:
            QMessageBox.warning(self, "VRAM check", warn)
        self.monitor_panel.reset()
        self.monitor_panel.set_benchmark_history(
            benchmark_store.load(default_base_dir(), p.name))
        self.monitor_panel.set_endpoints(
            p.settings.get("port", 8080),
            bool(p.settings.get("embeddings")),
            bool(p.settings.get("reranking")),
        )
        if p.runtime.detached:
            argv = build_command(p, detach=True)
            # Detached drops --rm, so a stale stopped container of this name
            # would block the run with "name already in use". Remove it first,
            # then chain the run (mirrors the router branch above). on_error
            # surfaces bad image / CDI / flag failures the terminal used to
            # show -- show_dialog is fixed here, at launch time, so a later
            # profile/mode switch before the error fires can't change it.
            self._spawn_async(
                runtime.rm_argv(self._container_name(), p.runtime.binary),
                on_done=lambda: self._spawn_async(
                    argv, on_done=self.update_status,
                    on_error=lambda e=None: self._report_launch_error(
                        e, show_dialog=True)))
        else:
            argv = build_command(p)
            terminal.launch(argv)
        # Don't attach the log follower here: the container is created
        # asynchronously and doesn't exist yet. update_status() starts the
        # follower once it is actually running (podman logs replays from the
        # start, so no early output is missed).

    def _spawn_async(self, argv: list[str], on_done=None, on_error=None):
        """Run argv in the background via QProcess so the UI thread never blocks.
        Calls on_done() when the process finishes.

        on_error(text) receives podman's stderr on a non-zero exit, and a
        message if the binary could not be started at all — `finished` is NOT
        emitted on FailedToStart, so without errorOccurred a launch that never
        began would report nothing whatsoever.
        """
        from PySide6.QtCore import QProcess
        proc = QProcess(self)
        if on_done is not None:
            proc.finished.connect(lambda *_: on_done())
        if on_error is not None:
            def _finished(code, _status):
                if code != 0:
                    err = bytes(proc.readAllStandardError()).decode(errors="replace").strip()
                    on_error(err or f"{argv[0]} exited {code}")
            proc.finished.connect(_finished)
            proc.errorOccurred.connect(
                lambda _e: on_error(f"could not run {argv[0]}: {proc.errorString()}"))
        proc.start(argv[0], argv[1:])
        return proc

    def on_stop(self):
        # Stop the log follower immediately; run `podman stop` asynchronously so a
        # slow stop (podman waits up to its grace period for SIGTERM) never freezes
        # the GUI — which previously made Stop look like it did nothing.
        self._stop_log_follower()
        p = self.current_profile()
        self.status_label.setText("● stopping…")
        argv = runtime.stop_argv(self._container_name(), p.runtime.binary)
        self._stop_proc = self._spawn_async(argv, on_done=self.update_status)

    def on_restart(self):
        self._stop_log_follower()
        p = self.current_profile()
        self.status_label.setText("● restarting…")
        argv = runtime.stop_argv(self._container_name(), p.runtime.binary)
        # Launch only after the stop completes, so the new container's --name/port
        # don't collide with the one being torn down.
        self._stop_proc = self._spawn_async(argv, on_done=self.on_launch)

    def _on_enable_metrics(self):
        self._widgets["metrics"].set_value(True)
        self.on_restart()

    def _resolve_benchmark_member(self, p: Profile, model_scope: str | None):
        """The child Profile for a router's loaded model, for build_snapshot().

        build_snapshot() requires a Profile (it reads .settings/.model), never
        a RouterMember -- a RouterMember has neither and raises AttributeError.
        Falling back to None (profile.settings, i.e. the router's own -- empty
        -- settings) is a lesser-quality snapshot but never crashes.
        """
        if p.mode != "router" or model_scope is None:
            return None
        for member, member_profile in self.member_pairs():
            if member_model_id(member) == model_scope:
                return member_profile
        return None

    def _prepare_benchmark(self, p: Profile):
        """Endpoint + snapshot + client for a benchmark run.

        Mirrors collect_monitor_data's host/port/key/model_scope derivation
        (main_window.py, same class) exactly. Returns (client, snapshot), or
        None when there's nothing to benchmark -- a router with no model
        currently loaded, the same "not ready" condition collect_monitor_data
        treats as poll=False.
        """
        port = p.settings.get("port", 8080)
        host, key, model_scope, poll = dial_host(p.runtime.bind_host), None, None, True
        if p.mode == "router":
            host = self._router_host(p)
            key = api_key_store.read_api_key(self.router_base_dir(), p.name)
            model_scope = self._router_pollable_model()
            poll = model_scope is not None
        if not poll:
            return None
        member = self._resolve_benchmark_member(p, model_scope)
        snapshot = benchmark.build_snapshot(p, member=member)
        client = benchmark.requests_client(host, port, key, model_scope)
        return client, snapshot

    def _run_benchmark_sync(self, cfg: dict, run_benchmark=None) -> None:
        """Endpoint-build -> run_benchmark -> save/show, all on the calling thread.

        The testable seam: tests call this directly with a stubbed
        run_benchmark, so no QThread and no network round-trip is needed.
        `run_benchmark` resolves benchmark.run_benchmark dynamically when
        omitted (rather than as a plain default-argument value), so
        monkeypatching the module attribute after this method is defined
        still takes effect.
        """
        if run_benchmark is None:
            run_benchmark = benchmark.run_benchmark
        p = self.current_profile()
        prepared = self._prepare_benchmark(p)
        if prepared is None:
            self.monitor_panel.set_benchmark_progress("No model loaded to benchmark.")
            return
        client, snapshot = prepared
        self._benchmark_profile_name = p.name
        self.monitor_panel.set_benchmark_running(True)
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        try:
            run = run_benchmark(client, cfg["sizes"], cfg["n_predict"], cfg["warmup"],
                                cfg["repeats"], snapshot, timestamp)
        except benchmark.BenchmarkError as e:
            self._on_benchmark_failed(str(e))
            return
        self._on_benchmark_finished(run)

    def _on_benchmark_run(self, cfg: dict) -> None:
        """Production path: build the endpoint/client on the UI thread, then run
        benchmark.run_benchmark() on a QThread so a slow benchmark never blocks
        the GUI."""
        if self._benchmark_thread is not None:
            return          # a run is already active; the panel showed Cancel
        p = self.current_profile()
        prepared = self._prepare_benchmark(p)
        if prepared is None:
            self.monitor_panel.set_benchmark_progress("No model loaded to benchmark.")
            return
        client, snapshot = prepared
        self._benchmark_profile_name = p.name
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")

        thread = QThread(self)
        worker = BenchmarkWorker(client, cfg["sizes"], cfg["n_predict"], cfg["warmup"],
                                 cfg["repeats"], snapshot, timestamp)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # Queued across threads by Qt automatically -- worker lives on `thread`,
        # these slots run on the UI thread where GUI/store access is safe.
        worker.finished.connect(self._on_benchmark_finished)
        worker.failed.connect(self._on_benchmark_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._on_benchmark_thread_done)
        self._benchmark_thread = thread
        self._benchmark_worker = worker
        self.monitor_panel.set_benchmark_running(True)
        thread.start()

    def _on_benchmark_thread_done(self) -> None:
        """Release the finished thread/worker so a later run can start."""
        worker, thread = self._benchmark_worker, self._benchmark_thread
        self._benchmark_worker = None
        self._benchmark_thread = None
        if worker is not None:
            worker.deleteLater()
        if thread is not None:
            thread.deleteLater()

    def _on_benchmark_cancel(self) -> None:
        if self._benchmark_worker is not None:
            self._benchmark_worker.cancel()

    def _on_benchmark_finished(self, run) -> None:
        name = self._benchmark_profile_name or self.current_profile().name
        base = default_base_dir()
        previous_runs = benchmark_store.load(base, name)
        previous = previous_runs[-1] if previous_runs else None
        benchmark_store.append(base, name, run)
        run_dict = dataclasses.asdict(run)
        delta = benchmark_store.delta(run_dict, previous) if previous is not None else None
        self.monitor_panel.show_benchmark_run(run_dict, delta)
        self.monitor_panel.set_benchmark_history(benchmark_store.load(base, name))
        self.monitor_panel.set_benchmark_running(False)

    def _on_benchmark_failed(self, msg: str) -> None:
        self.monitor_panel.set_benchmark_progress(f"Benchmark failed: {msg}")
        self.monitor_panel.set_benchmark_running(False)

    def on_fetch_latest(self):
        repo, tag = split_image(self.image_edit.text())
        if not repo:
            return
        prefix = variant_prefix(tag) if tag else "server-cuda12"
        latest = registry.fetch_latest(repo, prefix)
        if latest:
            self.image_edit.setText(f"{repo}:{latest}")

    def detect_image(self):
        binary = self.binary_combo.currentText()
        images = runtime.list_local_images(binary)
        if not images:
            QMessageBox.information(
                self, "Detect image",
                f"No local llama.cpp images found for '{binary}'.\n"
                f"Pull one (e.g. {binary} pull ghcr.io/ggml-org/llama.cpp:server) "
                f"or type the image reference yourself.")
            return
        if len(images) == 1:
            self.image_edit.setText(images[0])
            return
        choice, ok = QInputDialog.getItem(
            self, "Detect image", "Local llama.cpp images:", images, 0, False)
        if ok and choice:
            self.image_edit.setText(choice)

    def _autofill_image_if_empty(self):
        if self.image_edit.text().strip():
            return
        images = runtime.list_local_images(self.binary_combo.currentText())
        if len(images) == 1:
            self.image_edit.setText(images[0])

    def check_for_update(self, tags: list[str]) -> str | None:
        repo, tag = split_image(self.image_edit.text())
        if not tag:
            return None
        prefix = variant_prefix(tag)
        latest = registry.latest_build_tag(tags, prefix)
        if latest and latest != tag:
            return latest
        return None

    def run_update_check(self):
        repo, tag = split_image(self.image_edit.text())
        if not repo or not tag:
            return
        prefix = variant_prefix(tag)
        current_tag = tag

        def _on_found(latest: str):
            if latest != current_tag:
                m = registry._BUILD_RE.match(latest)
                build_id = f"b{m.group('num')}" if m else latest
                self.update_badge.setText(f"newer build {build_id} available")
                self.update_badge.setVisible(True)

        worker = _UpdateWorker(repo, prefix, parent=self)
        worker.found.connect(_on_found)
        self._update_worker = worker
        worker.start()

    def update_status(self):
        p = self._monitored_profile()
        if not runtime.binary_available(p.runtime.binary):
            self.status_label.setText("● stopped")
            self.web_ui_btn.setEnabled(False)
            self.monitor_panel.set_benchmark_available(False)
            return
        name = self._monitored_container_name()
        state = runtime.container_state(name, p.runtime.binary)
        hstatus = health.probe_health(p.settings.get("port", 8080),
                                     host=dial_host(p.runtime.bind_host)) \
            if state == "running" else "down"
        self.status_label.setText("● " + health.derive_status(state, hstatus))
        self.web_ui_btn.setEnabled(state == "running")
        router_model_key = None
        if state == "running":
            if not self._log_follower_active():
                self._start_log_follower()
            if hstatus == "ready":
                router_model_key = self._refresh_props(p)
            self.monitor_panel.update_stats(self.collect_monitor_data())
            self._update_spec_stats(p)
        if p.mode == "router" and state == "running":
            self.refresh_router_models()
        # router_model_key was resolved from _refresh_props above (when ready)
        # rather than polled again here, so this reuses that single call to
        # _router_pollable_model() instead of doubling it.
        ready = state == "running" and hstatus == "ready"
        if ready and p.mode == "router":
            ready = router_model_key is not None
        self.monitor_panel.set_benchmark_available(ready)
        self._refresh_instances_list()

    def _refresh_instances_list(self) -> None:
        binary = self.current_profile().runtime.binary
        self._instances = build_instances(
            runtime.list_launcher_containers(binary), list_profiles(base_dir()))
        rows = []
        for inst in self._instances:
            summ = self.instance_summary(inst)
            rows.append({"name": inst.name, "profile": inst.profile, "port": inst.port,
                         "running": inst.running, "health": summ["health"],
                         "stat": summ["stat"]})
        self.monitor_panel.set_instances(rows, self._monitored_container_name())

    def _on_instance_selected(self, name: str) -> None:
        inst = next((i for i in self._instances if i.name == name), None)
        # Selecting the form's own container means "monitor the current profile" (fallback).
        self._active_instance = None if (inst is None or name == self._container_name()) else inst
        self._start_log_follower()          # retarget the follower at the new container
        self.update_status()

    def _on_instance_stop(self, name: str) -> None:
        binary = self.current_profile().runtime.binary
        self._spawn_async(runtime.stop_argv(name, binary), on_done=self.update_status)
        if self._active_instance is not None and self._active_instance.name == name:
            self._active_instance = None

    def _monitored_profile(self) -> Profile:
        inst = self._active_instance
        if inst is None:
            return self.current_profile()
        for prof in list_profiles(base_dir()):
            if prof.name == inst.profile:
                return prof
        return self.current_profile()      # profile deleted under us -> fall back

    def _monitored_container_name(self) -> str:
        return self._active_instance.name if self._active_instance else self._container_name()

    def instance_summary(self, inst) -> dict:
        if not inst.running or inst.port is None:
            return {"running": inst.running, "health": "down", "stat": ""}
        hstatus = health.probe_health(inst.port, host=inst.host)
        if inst.embeddings or inst.reranking:
            stat = "ready" if hstatus == "ready" else ""
        else:
            tok = metrics.fetch_metrics(inst.port, host=inst.host).get(
                "llamacpp:predicted_tokens_seconds")
            stat = f"{tok:.0f} tok/s" if tok else ("ready" if hstatus == "ready" else "")
        return {"running": True, "health": hstatus, "stat": stat}

    def collect_monitor_data(self) -> dict:
        from llama_launcher.services.metrics import kv_usage_ratio
        p = self._monitored_profile()
        port = p.settings.get("port", 8080)
        metrics_on = bool(p.settings.get("metrics"))
        host, key, model_scope, poll = dial_host(p.runtime.bind_host), None, None, True
        if p.mode == "router":
            host = self._router_host(p)
            key = api_key_store.read_api_key(self.router_base_dir(), p.name)
            model_scope = self._router_pollable_model()
            poll = model_scope is not None
        m = (metrics.fetch_metrics(port, model=model_scope, api_key=key, host=host)
             if metrics_on and poll else {})
        slots = (metrics.fetch_slots(port, model=model_scope, api_key=key, host=host)
                 if poll else [])
        name = self._monitored_container_name()
        st = runtime.stats(name, p.runtime.binary) or {}
        started = runtime.started_at(name, p.runtime.binary)
        uptime = _fmt_uptime(started)
        return {
            "tok_s": m.get("llamacpp:predicted_tokens_seconds"),
            "prompt_tok_s": m.get("llamacpp:prompt_tokens_seconds"),
            "kv_pct": kv_usage_ratio(slots),
            "speculating": any(s.get("speculative") for s in slots),
            "gpus": gpu.query_gpus(),
            "cpu": st.get("cpu_perc", ""),
            "mem": st.get("mem_usage", ""),
            "uptime": uptime,
            "metrics_on": metrics_on,
        }

    def _log_follower_active(self) -> bool:
        from PySide6.QtCore import QProcess
        return (self._log_proc is not None
                and self._log_proc.state() != QProcess.NotRunning)

    def _start_log_follower(self):
        from PySide6.QtCore import QProcess
        self._stop_log_follower()
        p = self._monitored_profile()
        name = self._monitored_container_name()
        # Attaching `podman logs -f` before the container exists just prints
        # "no such container" and exits, stranding the logs pane on that error.
        # Skip until it exists; update_status() retries once it's running and
        # `podman logs` replays from the beginning, so no early output is lost.
        if not runtime.container_exists(name, p.runtime.binary):
            return
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(
            lambda: self.monitor_panel.append_log(bytes(proc.readAllStandardOutput()).decode("utf-8", "replace")))
        argv = runtime.logs_argv(name, p.runtime.binary)
        proc.start(argv[0], argv[1:])
        self._log_proc = proc

    def _stop_log_follower(self):
        if self._log_proc is not None:
            self._log_proc.kill()
            self._log_proc = None

    def _stop_timers(self) -> None:
        """Stop background timers so a torn-down window stops firing update_status
        (and any pending update check). Idempotent — QTimer.stop() on a stopped
        timer is a no-op. _update_timer exists only when update_check is enabled.

        Also tears down a running benchmark thread: cancels the worker (so a
        loop mid-repeat unwinds instead of running to completion against a
        closed window) and waits for the QThread to actually stop, since a
        Python interpreter shutdown with a live QThread can abort/crash.

        worker.finished/failed are already wired to thread.quit() (see
        _on_benchmark_run), but that's a queued cross-thread connection: it
        only takes effect once THIS (UI) thread's event loop runs and
        delivers it. A bare wait() never pumps events, so it would deadlock
        waiting for a quit() that never arrives; pump events between short
        waits instead. terminate() is a last-resort backstop so a stuck
        worker can never block window/app teardown indefinitely.
        """
        self._status_timer.stop()
        update_timer = getattr(self, "_update_timer", None)
        if update_timer is not None:
            update_timer.stop()
        thread = getattr(self, "_benchmark_thread", None)
        if thread is None:
            return
        worker = getattr(self, "_benchmark_worker", None)
        if worker is not None:
            worker.cancel()
        from PySide6.QtCore import QCoreApplication
        for _ in range(100):        # ~2s ceiling
            if thread.wait(20):
                # One more pump so the already-queued finished/failed ->
                # _on_benchmark_thread_done delivery (which clears
                # _benchmark_thread/_benchmark_worker) actually runs before
                # we return, instead of leaving it dangling until whatever
                # next processes the event queue.
                QCoreApplication.processEvents()
                return
            QCoreApplication.processEvents()
        thread.terminate()
        thread.wait(100)

    def _on_export_sh(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export shell script", "run.sh",
                                              "Shell scripts (*.sh);;All files (*)")
        if path:
            self.export_sh(path)

    def open_web_ui(self):
        p = self.current_profile()
        port = p.settings.get("port", 8080)
        try:
            subprocess.Popen(["xdg-open", f"http://{dial_host(p.runtime.bind_host)}:{port}"],
                             start_new_session=True)
        except OSError:
            QMessageBox.warning(self, "Open Web UI", "Could not open browser (xdg-open not found).")

    def export_sh(self, path: str):
        cmd = " ".join(self.build_current_command())
        Path(path).write_text(f"#!/usr/bin/env bash\n{cmd}\n")
        os.chmod(path, 0o755)

    def gather_report_data(self) -> dict:
        import platform, json as _json
        p = self.current_profile()
        cmd = " ".join(self.build_current_command(p))
        # Pass the router context, or the report claims a healthy router has no
        # members and is exposed without a key -- in the one artifact users
        # paste when asking for help.
        issues = validate(p, binary_found=runtime.binary_available(p.runtime.binary),
                          members=self.member_pairs(),
                          api_key_present=bool(
                              api_key_store.read_api_key(self.router_base_dir(), p.name))
                          if p.mode == "router" else False)
        gpus = gpu.query_gpus()
        gpu_txt = "\n".join(f"{g.name}: {g.mem_used_mib}/{g.mem_total_mib} MiB, "
                            f"util {g.util_pct}%, {g.temp_c}C" for g in gpus) or "(no nvidia-smi)"
        runtime_txt = (f"binary={p.runtime.binary} gpu_mode={p.runtime.gpu_mode}\n"
                       f"rootless={runtime.is_rootless(p.runtime.binary)}\n"
                       f"{gpu_txt}\nOS={platform.platform()}")
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return {
            "generated_at": ts,
            "command": report_mod.redact_secrets(cmd),
            "profile": report_mod.redact_secrets(_json.dumps(profile_to_dict(p), indent=2)),
            "validation": [f"[{i.level}] {i.message}" for i in issues],
            "status_history": [self.status_label.text()],
            "runtime": runtime_txt,
            "metrics": self._metrics_report_text(p),
            "image": p.image,
            "logs": report_mod.redact_secrets(self.monitor_panel.log_view.toPlainText()[-4000:]),
        }

    def _metrics_report_text(self, p: Profile) -> str:
        """Snapshot of the live /metrics endpoint for the diagnostic report.

        Returns a note (and makes no network call) when --metrics is off, so the
        report explains why throughput is missing instead of silently omitting it.
        """
        from llama_launcher.services.metrics import kv_usage_ratio
        port = p.settings.get("port", 8080)
        if not p.settings.get("metrics"):
            return ("(--metrics not enabled in this profile — turn it on and relaunch "
                    "to capture tok/s and KV-cache usage here)")
        m = metrics.fetch_metrics(port)
        slots = metrics.fetch_slots(port)
        if not m and not slots:
            return (f"(no metrics returned from http://127.0.0.1:{port}/metrics — "
                    "generate the report while the server is running with --metrics)")
        lines = []
        gen = m.get("llamacpp:predicted_tokens_seconds")
        if gen is not None:
            lines.append(f"generation: {gen:.2f} tok/s")
        prompt = m.get("llamacpp:prompt_tokens_seconds")
        if prompt is not None:
            lines.append(f"prompt: {prompt:.2f} tok/s")
        kv = kv_usage_ratio(slots)
        if kv is not None:
            lines.append(f"KV cache usage: {kv * 100:.0f}%")
        if m:
            if lines:
                lines.append("")
            lines += [f"{k} {v:g}" for k, v in sorted(m.items())]
        return "\n".join(lines)

    def _save_report(self, md: str, ts: str | None = None) -> Path:
        if ts is None:
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        reports_dir = base_dir() / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        out = reports_dir / f"llama-launcher-report-{ts}.md"
        out.write_text(md)
        return out

    def on_generate_report(self):
        cfg = load_config(base_dir())
        initial = cfg.get("report_sections", {s: True for s in report_mod.REPORT_SECTIONS})
        dlg = ReportDialog(initial, self)
        if not dlg.exec():
            return
        sections = dlg.selected_sections()
        cfg["report_sections"] = sections
        save_config(cfg, base_dir())
        data = self.gather_report_data()
        md = report_mod.build_report(data, sections)
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(md)
        saved = self._save_report(md, data.get("generated_at"))
        QMessageBox.information(self, "Report saved", f"Report saved to:\n{saved}")

    def closeEvent(self, event):
        if getattr(self, "_really_quit", False) or not self._minimize_to_tray:
            self._stop_log_follower()
            self._stop_timers()
            event.accept()
            from PySide6.QtWidgets import QApplication
            QApplication.instance().quit()
        else:
            event.ignore()
            self.hide()

    def quit_app(self):
        self._really_quit = True
        self._stop_log_follower()
        self._stop_timers()
        from PySide6.QtWidgets import QApplication
        QApplication.instance().quit()

    def _set_field_relevance(self, edit, tier) -> None:
        edit.setProperty("relevance", getattr(tier, "value", str(tier)))
        edit.style().unpolish(edit)
        edit.style().polish(edit)

    def apply_model_caps(self) -> None:
        p = self.current_profile()
        meta, size, caps = (None, None, None)
        if p.model:
            meta, size, caps = model_info.inspect_model(p.model, self.mounts_panel.mounts())
        self._last_caps = caps
        self.model_meta_label.setText(self._meta_caps_text(meta, size, caps))
        tiers = relevance(caps) if caps else {}
        for key, widget in self._widgets.items():
            widget.set_relevance(tiers.get(key, Tier.NEUTRAL))
        self._set_field_relevance(self.mmproj_edit, tiers.get("mmproj", Tier.NEUTRAL))
        self._set_field_relevance(self.draft_model_edit, tiers.get("draft_model", Tier.NEUTRAL))
        self._rebuild_suggestions(caps)

    def _all_presets(self) -> list:
        # User presets override curated ones of the same key.
        merged = {p.key: p for p in PRESETS}
        for p in list_user_presets(base_dir()):
            merged[p.key] = p
        return list(merged.values())

    def _reload_family_combo(self) -> None:
        self.family_combo.clear()
        self.family_combo.addItem("(none)", None)
        for preset in self._all_presets():
            self.family_combo.addItem(preset.label, preset)

    def _on_pick_family(self, _index) -> None:
        self._preset_family = self.family_combo.currentData()
        self._rebuild_suggestions(self._last_caps)

    def on_save_preset(self) -> None:
        active = self.active_catalog()
        captured = {k: self._widgets[k].value()
                    for k in active if self._widgets[k].is_set()}
        captured.pop("port", None)                 # port is machine state, not a family flag
        if not captured:
            QMessageBox.information(
                self, "Nothing to save",
                "Set some options first — a preset captures the options you've set.")
            return
        name, ok = QInputDialog.getText(self, "Save as preset", "Preset name:")
        if not ok or not name.strip():
            return
        preset = Preset(key=slugify(name), label=name.strip(),
                        settings=captured, source="user")
        save_user_preset(preset, base_dir())
        self._reload_family_combo()
        idx = self.family_combo.findText(preset.label)
        if idx >= 0:
            self.family_combo.setCurrentIndex(idx)
            self._on_pick_family(idx)

    def _rebuild_suggestions(self, caps) -> None:
        while self._suggestions_layout.count():
            item = self._suggestions_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        sgs = []
        if caps:
            sgs += compute_suggestions(
                caps, self.current_profile().settings,
                mmproj_set=bool(self.mmproj_edit.text()),
                draft_set=bool(self.draft_model_edit.text()),
            )
        if self._preset_family is not None:
            active = self.active_catalog()
            sgs += [s for s in preset_suggestions(self._preset_family)
                    if all(k in active for k in s.settings)]
        for sg in sgs:
            btn = QPushButton("💡 " + sg.text)
            btn.setFlat(True)
            btn.clicked.connect(lambda _=False, s=sg: self._apply_suggestion(s))
            self._suggestions_layout.addWidget(btn)
        self._suggestions_layout.addStretch(1)

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
