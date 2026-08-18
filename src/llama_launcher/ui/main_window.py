import dataclasses
import os
import subprocess
from pathlib import Path

import datetime

from PySide6.QtCore import Qt, QObject, QThread, Signal
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit,
    QCheckBox, QGroupBox, QScrollArea, QLabel, QPlainTextEdit, QPushButton,
    QMessageBox, QFileDialog, QInputDialog, QTabWidget, QDockWidget
)

from llama_launcher.core.spec import (
    Profile, Mount, RouterMember, member_model_id, slugify,
)
from llama_launcher.core.router_preset import render_preset
from llama_launcher.core.command_builder import build_command
from llama_launcher.core.pathmap import host_to_container
from llama_launcher.core.validation import (
    validate, LOOPBACK_HOSTS, dial_host,
)
from llama_launcher.store.profiles import (
    default_base_dir, list_profiles, save_profile, delete_profile,
    load_config, save_config, profile_to_dict,
)
from llama_launcher.core.instances import Instance
# health and router_api are no longer called directly in this file (that
# behavior moved to MonitorController, ui/controllers/monitor_controller.py)
# but stay imported here too: the test suite monkeypatches them as
# `mw.health.probe_health` / `llama_launcher.ui.main_window.router_api.*`,
# and both names resolve to the same module objects monitor_controller.py's
# own imports use, so the patch still reaches the real call sites.
from llama_launcher.services import runtime, terminal, registry, health, metrics, gpu, model_info
from llama_launcher.services import benchmark, benchmark_store
from llama_launcher.core import vram
from llama_launcher.core import report as report_mod
from llama_launcher.services.registry import split_image, variant_prefix
from llama_launcher.ui.dialogs.report_dialog import ReportDialog
from llama_launcher.ui.widgets.setting_widgets import make_widget, SuggestionDot
from llama_launcher.ui.widgets.no_wheel import NoWheelComboBox
from llama_launcher.ui.panels.mounts_panel import MountsPanel
from llama_launcher.ui.panels.lora_panel import LoraPanel
from llama_launcher.ui.panels.configure_panel import ConfigurePanel
from llama_launcher.ui.widgets.collapsible import CollapsibleSection
from llama_launcher.ui.panels.monitor_panel import MonitorPanel
from llama_launcher.ui.panels.benchmark_panel import BenchmarkPanel
from llama_launcher.ui.panels.stats_panel import StatsPanel
from llama_launcher.ui.widgets.api_key_box import ApiKeyBox
from llama_launcher.ui.widgets.harness_info_box import HarnessInfoBox
from llama_launcher.ui.widgets.router_models_table import RouterModelsTable
from llama_launcher.ui.widgets.status_banner import StatusBanner
from llama_launcher.services import api_key as api_key_store
from llama_launcher.services import router_api


class _UpdateWorker(QThread):
    found = Signal(str)
    failed = Signal(str)

    def __init__(self, repo: str, prefix: str, parent=None):
        super().__init__(parent)
        self._repo = repo
        self._prefix = prefix

    def run(self):
        try:
            tag = registry.fetch_latest(self._repo, self._prefix)
            if tag:
                self.found.emit(tag)
        except Exception as e:            # noqa: BLE001 - surfaced to the user
            self.failed.emit(str(e))


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


# StatsWorker/_MonitorGather/build_monitor_data/_fmt_uptime used to be defined
# in this module; they now live in monitor_controller.py (moved along with the
# status/instances/monitor/log-follower/stats/router-poll behavior that uses
# them) but are re-exported here too, since the test suite still reaches them
# as `llama_launcher.ui.main_window.StatsWorker` /
# `llama_launcher.ui.main_window.build_monitor_data`.
#
# This import is placed after base_dir() (which monitor_controller.py imports
# back, lazily, per-method -- see its local `from llama_launcher.ui.main_window
# import base_dir` calls) so that name is already bound on this module by the
# time those calls resolve.
from llama_launcher.ui.controllers.monitor_controller import (  # noqa: E402
    MonitorController, StatsWorker, _MonitorGather, build_monitor_data, _fmt_uptime,
)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Llama Launcher")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        self._configure_panel = ConfigurePanel(self)
        # Owns status/instances/monitor/log-follower/stats/router-poll behavior
        # (see monitor_controller.py) and the state it reads/writes between
        # ticks. Built here -- before any widget/signal wiring below -- so
        # that state exists from the very start of __init__, ahead of any
        # signal that could synchronously reach it.
        self._monitor = MonitorController(self)

        self._benchmark_thread = None
        self._benchmark_worker = None
        self._benchmark_profile_name = None

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self._configure_panel.configure_tab, "Configure")
        self.monitor_panel = MonitorPanel()
        self.monitor_panel.enable_metrics_requested.connect(self._on_enable_metrics)
        self.monitor_panel.instance_selected.connect(self._monitor._on_instance_selected)
        self.monitor_panel.instance_stop_requested.connect(self._monitor._on_instance_stop)
        self.monitor_panel.instance_remove_requested.connect(self._monitor._on_instance_remove)
        # Scroll the Monitor tab (like Configure): a short window otherwise
        # squeezes the log to a few lines. The log now owns the tab (benchmark
        # moved to its own tab), so it fills the height.
        monitor_scroll = QScrollArea()
        monitor_scroll.setWidgetResizable(True)
        monitor_scroll.setWidget(self.monitor_panel)
        self.tabs.addTab(monitor_scroll, "Monitor")

        self.router_models_table = RouterModelsTable()
        self.router_models_table.load_requested.connect(self._monitor._on_router_load)
        self.router_models_table.unload_requested.connect(self._monitor._on_router_unload)
        self.monitor_status = StatusBanner()
        self.monitor_panel.add_status_banner(self.monitor_status)
        self.monitor_panel.add_below_log(self.router_models_table)

        self.benchmark_panel = BenchmarkPanel()
        self.benchmark_panel.benchmark_run_requested.connect(self._on_benchmark_run)
        self.benchmark_panel.benchmark_cancel_requested.connect(self._on_benchmark_cancel)
        self.benchmark_panel.benchmark_clear_requested.connect(self._on_benchmark_clear)
        self.tabs.addTab(self.benchmark_panel, "Benchmark")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        # Stretch=1 so the tab body (Configure's Environment/Settings columns)
        # fills the vertical space down to the command-preview strip, rather
        # than leaving dead space where the router-only widgets are hidden.
        root.addWidget(self.tabs, 1)

        # Right-hand stats dock (GPU / system / container). Hidden by default;
        # a checkable button in the top bar toggles it. Polling is wired in the
        # StatsWorker task and runs only while the dock is visible.
        self.stats_panel = StatsPanel()
        self.stats_dock = QDockWidget("Stats", self)
        self.stats_dock.setObjectName("stats_dock")
        self.stats_dock.setWidget(self.stats_panel)
        self.stats_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, self.stats_dock)
        self.stats_dock.hide()
        # _monitor (built above, ahead of all widget/signal wiring) already
        # owns _stats_worker/_cpu_sampler/_stats_target/_active_instance, so
        # the visibility/toggle signals wired immediately below -- and the
        # stats_open restore later in __init__, which programmatically checks
        # the toggle button -- can safely reach _start_stats_worker() straight
        # away.
        self.stats_dock.visibilityChanged.connect(self._monitor._on_stats_visibility)

        # BOTTOM: command preview is config-only; wrap it in one container so
        # it can be hidden on the Monitor/Router/Benchmark tabs (see
        # _on_tab_changed). Launch/Stop/etc stay shared below.
        root.addWidget(self._config_bottom)
        buttons = QHBoxLayout()
        for b in (self.launch_btn, self.stop_btn, self.restart_btn, self.web_ui_btn):
            buttons.addWidget(b)
        buttons.addWidget(self.detached_check)
        root.addLayout(buttons)

        # profile bar (added to the top of root via insertLayout)
        bar = QHBoxLayout()
        self.stats_toggle_btn = QPushButton("📊 Stats")
        self.stats_toggle_btn.setCheckable(True)
        self.stats_toggle_btn.setToolTip("Show/hide the live stats panel (Ctrl+Shift+S)")
        self.status_label = QLabel("● stopped")
        bar.addWidget(QLabel("Name"))
        bar.addWidget(self.name_edit, 1)
        bar.addWidget(self.profile_combo, 1)
        for b in (self.save_btn, self.save_as_btn, self.delete_btn, self.report_btn,
                  self.stats_toggle_btn):
            bar.addWidget(b)
        bar.addWidget(self.status_label)
        root.insertLayout(0, bar)
        self.stats_toggle_btn.toggled.connect(self.stats_dock.setVisible)
        # Also drive worker start/stop straight off the toggle: QDockWidget's
        # visibilityChanged only fires once the top-level window has actually
        # been shown (isVisible() requires the whole ancestor chain mapped),
        # which never happens for an unshown/offscreen MainWindow -- e.g. in
        # tests, or if stats are toggled before the first show(). toggled()
        # fires synchronously regardless. _on_stats_visibility's start/stop
        # calls are idempotent, so double-firing (this + visibilityChanged,
        # once the window is shown) is harmless.
        self.stats_toggle_btn.toggled.connect(self._monitor._on_stats_visibility)
        from PySide6.QtGui import QShortcut, QKeySequence
        QShortcut(QKeySequence("Ctrl+Shift+S"), self,
                  activated=lambda: self.stats_toggle_btn.toggle())

        self._reload_profile_list()

        # Apply the initial mode's visibility. The combo defaults to "server"
        # and startup loads no profile, so currentIndexChanged never fires --
        # without this, the router-only member widgets (created visible) show on
        # the default server-mode form until the user flips the mode combo.
        self._on_mode_changed()

        self.refresh_preview()

        _stats_cfg = load_config(base_dir())
        if _stats_cfg.get("stats_open", False):
            self.stats_toggle_btn.setChecked(True)     # shows the dock
        _w = int(_stats_cfg.get("stats_width", 320) or 320)
        self.resizeDocks([self.stats_dock], [_w], Qt.Horizontal)

        self._stop_proc = None
        self._fetch_worker = None
        self._update_worker = None

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
        self._status_timer.timeout.connect(self._monitor.update_status)
        self._status_timer.start()

        if load_config(base_dir()).get("update_check", True):
            self._update_timer = QTimer(self)
            self._update_timer.setSingleShot(True)
            self._update_timer.timeout.connect(self.run_update_check)
            self._update_timer.start(3000)

    # -- MonitorController state forwarders ----------------------------------
    # This state now lives on self._monitor (see ui/controllers/monitor_
    # controller.py); these r/w properties keep it reachable as window.<name>
    # for MainWindow's own not-moved code (e.g. on_launch's post-launch reset,
    # ConfigurePanel.load_profile), for _MonitorGather (which still writes
    # _monitor_result/_monitor_inflight onto the window it was handed), and
    # for the test suite.
    @property
    def _active_instance(self):
        return self._monitor._active_instance

    @_active_instance.setter
    def _active_instance(self, value):
        self._monitor._active_instance = value

    @property
    def _monitor_target(self):
        return self._monitor._monitor_target

    @_monitor_target.setter
    def _monitor_target(self, value):
        self._monitor._monitor_target = value

    @property
    def _monitor_result(self):
        return self._monitor._monitor_result

    @_monitor_result.setter
    def _monitor_result(self, value):
        self._monitor._monitor_result = value

    @property
    def _monitor_inflight(self):
        return self._monitor._monitor_inflight

    @_monitor_inflight.setter
    def _monitor_inflight(self, value):
        self._monitor._monitor_inflight = value

    @property
    def _stats_worker(self):
        return self._monitor._stats_worker

    @_stats_worker.setter
    def _stats_worker(self, value):
        self._monitor._stats_worker = value

    @property
    def _stats_target(self):
        return self._monitor._stats_target

    @_stats_target.setter
    def _stats_target(self, value):
        self._monitor._stats_target = value

    @property
    def _log_proc(self):
        return self._monitor._log_proc

    @_log_proc.setter
    def _log_proc(self, value):
        self._monitor._log_proc = value

    @property
    def _props(self):
        return self._monitor._props

    @_props.setter
    def _props(self, value):
        self._monitor._props = value

    @property
    def _props_model(self):
        return self._monitor._props_model

    @_props_model.setter
    def _props_model(self, value):
        self._monitor._props_model = value

    @property
    def _spec_prev(self):
        return self._monitor._spec_prev

    @_spec_prev.setter
    def _spec_prev(self, value):
        self._monitor._spec_prev = value

    @property
    def _router_statuses(self):
        return self._monitor._router_statuses

    @_router_statuses.setter
    def _router_statuses(self, value):
        self._monitor._router_statuses = value

    # -- ConfigurePanel widget forwarders -----------------------------------
    # Every Configure-tab widget now lives on self._configure_panel; these
    # properties keep it reachable as window.<name> for the rest of
    # MainWindow's methods (not yet moved) and for the test suite.
    @property
    def name_edit(self):
        return self._configure_panel.name_edit

    @property
    def image_edit(self):
        return self._configure_panel.image_edit

    @property
    def model_edit(self):
        return self._configure_panel.model_edit

    @property
    def binary_combo(self):
        return self._configure_panel.binary_combo

    @property
    def gpu_combo(self):
        return self._configure_panel.gpu_combo

    @property
    def engine_combo(self):
        return self._configure_panel.engine_combo

    @property
    def mode_combo(self):
        return self._configure_panel.mode_combo

    @property
    def bind_host_combo(self):
        return self._configure_panel.bind_host_combo

    @property
    def detached_check(self):
        return self._configure_panel.detached_check

    @property
    def selinux_check(self):
        return self._configure_panel.selinux_check

    @property
    def extra_args_edit(self):
        return self._configure_panel.extra_args_edit

    @property
    def raw_edit(self):
        return self._configure_panel.raw_edit

    @property
    def mmproj_edit(self):
        return self._configure_panel.mmproj_edit

    @property
    def draft_model_edit(self):
        return self._configure_panel.draft_model_edit

    @property
    def mounts_panel(self):
        return self._configure_panel.mounts_panel

    @property
    def lora_panel(self):
        return self._configure_panel.lora_panel

    @property
    def lora_section(self):
        return self._configure_panel.lora_section

    @property
    def model_meta_label(self):
        return self._configure_panel.model_meta_label

    @property
    def _draft_model_dot(self):
        return self._configure_panel._draft_model_dot

    @property
    def _mmproj_dot(self):
        return self._configure_panel._mmproj_dot

    @property
    def _left_form(self):
        return self._configure_panel._left_form

    @property
    def add_member_btn(self):
        return self._configure_panel.add_member_btn

    @property
    def remove_member_btn(self):
        return self._configure_panel.remove_member_btn

    @property
    def edit_member_btn(self):
        return self._configure_panel.edit_member_btn

    @property
    def members_list(self):
        return self._configure_panel.members_list

    @property
    def members_guidance(self):
        return self._configure_panel.members_guidance

    @property
    def _members_row(self):
        return self._configure_panel._members_row

    @property
    def profile_combo(self):
        return self._configure_panel.profile_combo

    @property
    def save_btn(self):
        return self._configure_panel.save_btn

    @property
    def save_as_btn(self):
        return self._configure_panel.save_as_btn

    @property
    def delete_btn(self):
        return self._configure_panel.delete_btn

    @property
    def launch_btn(self):
        return self._configure_panel.launch_btn

    @property
    def stop_btn(self):
        return self._configure_panel.stop_btn

    @property
    def restart_btn(self):
        return self._configure_panel.restart_btn

    @property
    def detect_image_btn(self):
        return self._configure_panel.detect_image_btn

    @property
    def fetch_btn(self):
        return self._configure_panel.fetch_btn

    @property
    def export_sh_btn(self):
        return self._configure_panel.export_sh_btn

    @property
    def report_btn(self):
        return self._configure_panel.report_btn

    @property
    def web_ui_btn(self):
        return self._configure_panel.web_ui_btn

    @property
    def api_key_box(self):
        return self._configure_panel.api_key_box

    @property
    def harness_box(self):
        return self._configure_panel.harness_box

    @property
    def configure_status(self):
        return self._configure_panel.configure_status

    @property
    def preview(self):
        return self._configure_panel.preview

    @property
    def _config_bottom(self):
        return self._configure_panel._config_bottom

    @property
    def _widgets(self):
        return self._configure_panel._widgets

    @property
    def _group_boxes(self):
        return self._configure_panel._group_boxes

    @property
    def _setting_rows(self):
        return self._configure_panel._setting_rows

    @property
    def update_badge(self):
        return self._configure_panel.update_badge

    @property
    def configure_tab(self):
        return self._configure_panel.configure_tab

    @property
    def _configure_tab(self):
        return self._configure_panel.configure_tab

    # -- ConfigurePanel behavior delegators ----------------------------------
    # These methods now live on self._configure_panel (Task 2 of the
    # main_window decomposition); MainWindow keeps a one-line forwarder for
    # each so both the test suite and MainWindow's own not-yet-moved code
    # (which calls them as self.<method>()) keep working unchanged.
    def _profile_name(self) -> str:
        return self._configure_panel._profile_name()

    def _container_name(self) -> str:
        return f"llama-{slugify(self._profile_name())}"

    def active_catalog(self) -> dict:
        return self._configure_panel.active_catalog()

    def _apply_mode_to_settings_form(self) -> None:
        return self._configure_panel._apply_mode_to_settings_form()

    def _on_mode_changed(self, _index=0) -> None:
        return self._configure_panel._on_mode_changed(_index)

    def _apply_engine_enums(self) -> None:
        return self._configure_panel._apply_engine_enums()

    def _maybe_seed_default_image(self, engine: str) -> None:
        return self._configure_panel._maybe_seed_default_image(engine)

    def _on_engine_changed(self, _index=0) -> None:
        return self._configure_panel._on_engine_changed(_index)

    def _sync_load_mode_legacy(self) -> None:
        return self._configure_panel._sync_load_mode_legacy()

    def _on_tab_changed(self, _index: int) -> None:
        # Entering the Configure tab must show a live key even for an edited-
        # but-unsaved router; refresh_router_panel_header() no-ops for
        # non-routers.
        if self.tabs.currentWidget() is self._configure_tab:
            self.refresh_router_panel_header()
        # Command preview / api-key / harness only make sense while configuring,
        # so hide the bottom strip on the Monitor/Benchmark tabs.
        self._config_bottom.setVisible(self.tabs.currentWidget() is self._configure_tab)

    # -- MonitorController delegators (status/instances/monitor/log/stats/
    # router-poll behavior now lives on self._monitor; see
    # ui/controllers/monitor_controller.py) --------------------------------
    def _on_stats_visibility(self, visible: bool) -> None:
        return self._monitor._on_stats_visibility(visible)

    def _refresh_stats_target(self) -> None:
        return self._monitor._refresh_stats_target()

    def _start_stats_worker(self) -> None:
        return self._monitor._start_stats_worker()

    def _stop_stats_worker(self) -> None:
        return self._monitor._stop_stats_worker()

    def _save_stats_config(self) -> None:
        return self._monitor._save_stats_config()

    def _add_member_item(self, member: RouterMember) -> None:
        return self._configure_panel._add_member_item(member)

    def set_member_fields(self, row: int, model_id: str | None = None,
                          load_on_startup: bool | None = None,
                          stop_timeout: int | None = None) -> None:
        return self._configure_panel.set_member_fields(
            row, model_id=model_id, load_on_startup=load_on_startup,
            stop_timeout=stop_timeout)

    def _member_candidates(self) -> list[str]:
        return self._configure_panel._member_candidates()

    def _on_add_member(self) -> None:
        return self._configure_panel._on_add_member()

    def _on_remove_member(self) -> None:
        return self._configure_panel._on_remove_member()

    def _has_unsaved_changes(self) -> bool:
        return self._configure_panel._has_unsaved_changes()

    def _on_edit_member(self) -> None:
        return self._configure_panel._on_edit_member()

    def members(self) -> list:
        return self._configure_panel.members()

    def member_pairs(self) -> list:
        return self._configure_panel.member_pairs()

    def missing_member_profiles(self) -> list:
        return self._configure_panel.missing_member_profiles()

    def router_issues(self) -> list:
        return self._configure_panel.router_issues()

    def _field_with_browse(self, line_edit: QLineEdit, dot: QWidget | None = None) -> QWidget:
        return self._configure_panel._field_with_browse(line_edit, dot)

    def _browse_into(self, line_edit: QLineEdit) -> None:
        return self._configure_panel._browse_into(line_edit)

    def load_profile(self, p: Profile) -> None:
        return self._configure_panel.load_profile(p)

    def current_profile(self) -> Profile:
        return self._configure_panel.current_profile()

    def build_current_command(self, p: Profile | None = None) -> list:
        return self._configure_panel.build_current_command(p)

    def preview_text(self) -> str:
        return self._configure_panel.preview_text()

    def refresh_preview(self) -> None:
        return self._configure_panel.refresh_preview()

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
        self._configure_panel._profile = p
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
        p = self.current_profile()
        return (api_key_store.resolve_api_key(self.router_base_dir(), p)
                or api_key_store.ensure_api_key(self.router_base_dir(), p.name))

    def prepare_router_files(self) -> tuple:
        """Write models.ini + api-key for the current router. Returns (dir, warnings)."""
        name = self._profile_name()
        result = render_preset(self.member_pairs())
        api_key_store.prepare_launch_key(self.router_base_dir(), self.current_profile())
        api_key_store.write_preset(self.router_base_dir(), name, result.text)
        return str(api_key_store.router_dir(self.router_base_dir(), name)), result.warnings

    def _on_key_scope_changed(self, mode: str) -> None:
        # The radio already feeds current_profile(); persist and re-resolve display.
        self.save_current_profile()
        self.refresh_router_panel_header()
        self._notify_key_change_needs_relaunch()

    def _on_key_saved(self, scope: str, value: str) -> None:
        base = self.router_base_dir()
        if scope == "global":
            api_key_store.write_global_key(base, value)
        else:
            api_key_store.set_profile_key(base, self._profile_name(), value)
        self.refresh_router_panel_header()
        self._notify_key_change_needs_relaunch()

    def _notify_key_change_needs_relaunch(self) -> None:
        """The running router keeps serving the key it launched with -- a key
        change here only takes effect on the NEXT launch. Without this, a user
        who copies the newly-shown key into their harness while the router is
        still up gets a bare 401 with nothing in the GUI explaining why."""
        p = self.current_profile()
        if runtime.container_state(self._container_name(),
                                   p.runtime.binary) == "running":
            QMessageBox.information(
                self, "Relaunch needed",
                "The router is running. Relaunch it for the new API key to "
                "take effect.")

    def _set_router_connected(self, connected: bool) -> None:
        self.configure_status.set_connected(connected)
        self.monitor_status.set_connected(connected)

    def _set_router_error(self, text: str) -> None:
        self.configure_status.set_error(text)
        self.monitor_status.set_error(text)

    def _set_router_exposure(self, text: str) -> None:
        self.configure_status.set_exposure_warning(text)
        self.monitor_status.set_exposure_warning(text)

    def refresh_router_panel_header(self) -> None:
        p = self.current_profile()
        if p.mode != "router":
            # Clear relocated router state so a previous router's exposure
            # banner, API key, and harness endpoint don't linger on the
            # Configure/Monitor tabs after switching to an unrelated profile.
            self._set_router_exposure("")
            self.api_key_box.set_key("")
            self.harness_box.harness_text.setPlainText("")
            return
        host = p.runtime.bind_host
        display_host = dial_host(host)
        port = p.settings.get("port", 8080)
        # A router without a key is unusable, and the harness block exists so the
        # key can be copied BEFORE the first launch. Generating here is
        # idempotent and is the only side effect on this path.
        key = (api_key_store.resolve_api_key(self.router_base_dir(), p)
               or api_key_store.ensure_api_key(self.router_base_dir(), p.name))
        self.api_key_box.set_key(key)
        self.harness_box.set_endpoint(
            f"http://{display_host}:{port}",
            [member_model_id(m) for m in p.members])
        self._set_router_exposure(
            f"Bound to {host}: reachable beyond this machine. The API key is required."
            if host not in LOOPBACK_HOSTS else "")

    def _update_spec_stats(self, p: Profile) -> None:
        return self._monitor._update_spec_stats(p)

    def _poll_api_key(self, p: Profile) -> str | None:
        return self._monitor._poll_api_key(p)

    def _refresh_props(self, p: Profile) -> str | None:
        return self._monitor._refresh_props(p)

    def _report_launch_error(self, text: str = None, *, show_dialog: bool = False) -> None:
        """Show why a detached launch -- router or server -- failed to start.
        Routed to the status banners (non-modal: this fires from a QProcess
        signal, which tests drive); a detached SERVER launch also pops a
        QMessageBox, since a Monitor-tab-only user may never see the
        Configure tab's banner and would otherwise miss the failure entirely.

        `show_dialog` is decided by the CALLER at launch time (when the
        profile's mode is known synchronously), not re-derived here from
        live UI state: this fires from an async QProcess callback, possibly
        seconds later, by which point the user may have switched profiles
        or flipped the mode combo -- current_profile() at that moment would
        no longer describe the launch that actually failed."""
        self.status_label.setText("● failed to start")
        reason = (f"launch failed: {text.splitlines()[-1][:200]}"
                  if text else "launch failed")
        self._set_router_error(reason)
        if show_dialog:
            QMessageBox.critical(self, "Launch failed", reason)

    def adopt_running_containers(self) -> list:
        """Containers this launcher owns, so a detached router survives a GUI restart."""
        p = self.current_profile()
        return runtime.list_launcher_containers(p.runtime.binary)

    def _router_host(self, p: Profile) -> str:
        return self._monitor._router_host(p)

    def refresh_router_models(self) -> None:
        return self._monitor.refresh_router_models()

    def _router_pollable_model(self) -> str | None:
        return self._monitor._router_pollable_model()

    def _on_router_load(self, model_id: str) -> None:
        return self._monitor._on_router_load(model_id)

    def _on_router_unload(self, model_id: str) -> None:
        return self._monitor._on_router_unload(model_id)

    def _validate_or_warn(self) -> bool:
        p = self.current_profile()
        # The key must exist before the exposure rule is evaluated: a router
        # always gets one at launch, but this runs before prepare_router_files.
        if p.mode == "router":
            api_key_store.prepare_launch_key(self.router_base_dir(), p)
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
        gpus = gpu.query_gpus()
        if meta is None or not gpus or not meta.n_layers or not meta.n_embd:
            return None
        mib = 1024 * 1024
        free_per_gpu = [g.mem_free_mib * mib for g in gpus]
        # Budget depends on how the model is placed: split across all GPUs (the
        # default) means their combined free VRAM; split-mode none means one card.
        split_mode = p.settings.get("split-mode", "layer")
        main_gpu = p.settings.get("main-gpu", 0)
        free = vram.available_free_bytes(free_per_gpu, split_mode, main_gpu)
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
        # Show the per-GPU breakdown when the budget spans multiple cards, so the
        # "free" number is transparent (e.g. "14.7 + 7.3 = 22.0 GiB across 2 GPUs").
        if len(free_per_gpu) > 1 and split_mode != "none":
            parts = " + ".join(f"{b/gib:.1f}" for b in free_per_gpu)
            free_txt = f"~{free/gib:.1f} GiB ({parts} across {len(free_per_gpu)} GPUs)"
        else:
            free_txt = f"~{free/gib:.1f} GiB"
        return (f"Estimated VRAM need ~{est.total_bytes/gib:.1f} GiB exceeds free "
                f"{free_txt} by ~{-margin/gib:.1f} GiB. It may not fit — "
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
            self.benchmark_panel.reset()
            self.benchmark_panel.set_benchmark_history(
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
        self.benchmark_panel.reset()
        self.benchmark_panel.set_benchmark_history(
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
        host, key, model_scope, poll = (dial_host(p.runtime.bind_host),
                                        self._poll_api_key(p), None, True)
        if p.mode == "router":
            host = self._router_host(p)
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
            self.benchmark_panel.set_benchmark_progress("No model loaded to benchmark.")
            return
        client, snapshot = prepared
        self._benchmark_profile_name = p.name
        self.benchmark_panel.set_benchmark_running(True)
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
            self.benchmark_panel.set_benchmark_progress("No model loaded to benchmark.")
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
        self.benchmark_panel.set_benchmark_running(True)
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

    def _on_benchmark_clear(self) -> None:
        """Wipe the saved benchmark history for the current profile and the view."""
        benchmark_store.clear(default_base_dir(), self.current_profile().name)
        self.benchmark_panel.reset()

    def _on_benchmark_finished(self, run) -> None:
        name = self._benchmark_profile_name or self.current_profile().name
        base = default_base_dir()
        previous_runs = benchmark_store.load(base, name)
        previous = previous_runs[-1] if previous_runs else None
        benchmark_store.append(base, name, run)
        run_dict = dataclasses.asdict(run)
        delta = benchmark_store.delta(run_dict, previous) if previous is not None else None
        self.benchmark_panel.show_benchmark_run(run_dict, delta)
        self.benchmark_panel.set_benchmark_history(benchmark_store.load(base, name))
        self.benchmark_panel.set_benchmark_running(False)

    def _on_benchmark_failed(self, msg: str) -> None:
        self.benchmark_panel.set_benchmark_progress(f"Benchmark failed: {msg}")
        self.benchmark_panel.set_benchmark_running(False)

    def on_fetch_latest(self):
        repo, tag = split_image(self.image_edit.text())
        if not repo:
            QMessageBox.information(
                self, "No image",
                "Set or Detect an image first — Fetch latest looks up the newest "
                "build for the image's repository.")
            return
        prefix = variant_prefix(tag) if tag else "server-cuda12"
        self._fetch_repo = repo
        self._fetch_got_result = False
        self.fetch_btn.setEnabled(False)
        self.fetch_btn.setText("Fetching…")
        self.update_badge.setEnabled(False)
        worker = _UpdateWorker(repo, prefix, parent=self)
        worker.found.connect(self._on_fetch_found)
        worker.failed.connect(self._on_fetch_failed)
        worker.finished.connect(self._on_fetch_finished)   # QThread built-in
        self._fetch_worker = worker
        worker.start()

    def _on_fetch_found(self, tag: str) -> None:
        self._fetch_got_result = True
        image = f"{self._fetch_repo}:{tag}"
        self.image_edit.setText(image)
        QMessageBox.information(
            self, "Latest build",
            f"Image set to {image}.\n\nThis only updates the tag — the build is NOT "
            f"downloaded. Pull it with:\n  podman pull {image}\n(or docker pull).")

    def _on_fetch_failed(self, msg: str) -> None:
        self._fetch_got_result = True
        QMessageBox.warning(
            self, "Fetch failed", f"Couldn't fetch the latest build:\n{msg}")

    def _on_fetch_finished(self) -> None:
        self.fetch_btn.setEnabled(True)
        self.fetch_btn.setText("Fetch latest")
        self.update_badge.setEnabled(True)
        if not self._fetch_got_result:
            QMessageBox.information(
                self, "Latest build", "No newer build found for this image.")

    def detect_image(self):
        binary = self.binary_combo.currentText()
        engine = self.engine_combo.currentData() or "llama.cpp"
        images = runtime.list_local_images(binary, engine)
        if not images:
            example = ("ghcr.io/ikawrakow/ik-llama-cpp:cu12-server"
                       if engine == "ik_llama.cpp"
                       else "ghcr.io/ggml-org/llama.cpp:server")
            QMessageBox.information(
                self, "Detect image",
                f"No local {engine} images found for '{binary}'.\n"
                f"Pull one (e.g. {binary} pull {example}) or type the image yourself.")
            return
        if len(images) == 1:
            self.image_edit.setText(images[0])
            return
        choice, ok = QInputDialog.getItem(
            self, "Detect image", f"Local {engine} images:", images, 0, False)
        if ok and choice:
            self.image_edit.setText(choice)

    def _autofill_image_if_empty(self):
        if self.image_edit.text().strip():
            return
        images = runtime.list_local_images(
            self.binary_combo.currentText(), self.engine_combo.currentData() or "llama.cpp")
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
        return self._monitor.update_status()

    def _refresh_instances_list(self) -> None:
        return self._monitor._refresh_instances_list()

    def _on_instance_selected(self, name: str) -> None:
        return self._monitor._on_instance_selected(name)

    def _on_instance_stop(self, name: str) -> None:
        return self._monitor._on_instance_stop(name)

    def _on_instance_remove(self, name: str) -> None:
        return self._monitor._on_instance_remove(name)

    def _monitored_profile(self) -> Profile:
        return self._monitor._monitored_profile()

    def _monitored_container_name(self) -> str:
        return self._monitor._monitored_container_name()

    def _instance_api_key(self, inst) -> str | None:
        return self._monitor._instance_api_key(inst)

    def instance_summary(self, inst) -> dict:
        return self._monitor.instance_summary(inst)

    def _compute_monitor_target(self, running: bool, model_scope=None) -> dict:
        return self._monitor._compute_monitor_target(running, model_scope)

    def collect_monitor_data(self) -> dict:
        return self._monitor.collect_monitor_data()

    def _log_follower_active(self) -> bool:
        return self._monitor._log_follower_active()

    def _start_log_follower(self):
        return self._monitor._start_log_follower()

    def _enqueue_log(self, text: str) -> None:
        return self._monitor._enqueue_log(text)

    def _flush_log(self) -> None:
        return self._monitor._flush_log()

    def _stop_log_follower(self):
        return self._monitor._stop_log_follower()

    def _stop_timers(self) -> None:
        """Stop background timers so a torn-down window stops firing update_status
        (and any pending update check). Idempotent — QTimer.stop() on a stopped
        timer is a no-op. _update_timer exists only when update_check is enabled.
        Also drains the stats worker via _stop_stats_worker() so a torn-down
        window can't leave a StatsWorker QThread running.

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
        self._monitor.drain()

        # Drain any in-flight registry-fetch / update-check QThread so closing
        # the window mid-fetch can't destroy a running QThread (abort/crash).
        # _UpdateWorker.run() is a blocking network call with no cancel flag, so
        # wait with a ceiling and terminate() as a last-resort backstop.
        from PySide6.QtCore import QCoreApplication
        for _attr in ("_fetch_worker", "_update_worker"):
            w = getattr(self, _attr, None)
            if w is None or not w.isRunning():
                continue
            for _ in range(100):            # ~2s ceiling
                if w.wait(20):
                    break
                QCoreApplication.processEvents()
            else:
                w.terminate()
                w.wait(100)

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
                              api_key_store.resolve_api_key(self.router_base_dir(), p))
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
        from llama_launcher.services.metrics import kv_ratio
        port = p.settings.get("port", 8080)
        if not p.settings.get("metrics"):
            return ("(--metrics not enabled in this profile — turn it on and relaunch "
                    "to capture tok/s and KV-cache usage here)")
        # Mirror collect_monitor_data's host/key/scope derivation: /metrics needs
        # the API key, and on a router it is per-model (?model=id) reached via the
        # router host. Without these the report's fetch 401'd (or returned nothing)
        # and always printed the "no metrics returned" note for routers.
        host = dial_host(p.runtime.bind_host)
        key = self._poll_api_key(p)
        model_scope = None
        if p.mode == "router":
            host = self._router_host(p)
            model_scope = self._router_pollable_model()
        m = metrics.fetch_metrics(port, model=model_scope, api_key=key, host=host)
        slots = metrics.fetch_slots(port, model=model_scope, api_key=key, host=host)
        if not m and not slots:
            scope = " (no model currently loaded on the router)" if (
                p.mode == "router" and model_scope is None) else ""
            return (f"(no metrics returned from http://{host}:{port}/metrics{scope} — "
                    "generate the report while the server is running with --metrics)")
        lines = []
        gen = m.get("llamacpp:predicted_tokens_seconds")
        if gen is not None:
            lines.append(f"generation: {gen:.2f} tok/s")
        prompt = m.get("llamacpp:prompt_tokens_seconds")
        if prompt is not None:
            lines.append(f"prompt: {prompt:.2f} tok/s")
        kv = kv_ratio(m, slots)
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

    def apply_model_caps(self) -> None:
        return self._configure_panel.apply_model_caps()

    def _suggestion_index(self, caps):
        return self._configure_panel._suggestion_index(caps)

    @staticmethod
    def _dot_state_for(key, described, sugg_by_key, reason_by_key):
        from llama_launcher.ui.panels.configure_panel import ConfigurePanel
        return ConfigurePanel._dot_state_for(key, described, sugg_by_key, reason_by_key)

    def _apply_dot(self, widget, key, described, sugg_by_key, reason_by_key) -> None:
        return self._configure_panel._apply_dot(widget, key, described, sugg_by_key, reason_by_key)

    def _apply_field_dot(self, dot, key, described, sugg_by_key, reason_by_key) -> None:
        return self._configure_panel._apply_field_dot(dot, key, described, sugg_by_key, reason_by_key)

    def _resolve_sibling(self, filename) -> str | None:
        return self._configure_panel._resolve_sibling(filename)

    def _apply_suggestion(self, sg) -> None:
        return self._configure_panel._apply_suggestion(sg)

    @staticmethod
    def _meta_caps_text(meta, size, caps) -> str:
        from llama_launcher.ui.panels.configure_panel import ConfigurePanel
        return ConfigurePanel._meta_caps_text(meta, size, caps)
