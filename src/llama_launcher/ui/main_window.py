# subprocess is no longer called directly in this file (open_web_ui's
# subprocess.Popen call moved to ReportController,
# ui/controllers/report_controller.py) but stays imported here too: the test
# suite monkeypatches it as `mw.subprocess.Popen`, and both names resolve to
# the same module object report_controller.py's own import uses, so the patch
# still reaches the real call site.
import subprocess
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QScrollArea, QLabel, QPushButton,
    QMessageBox, QInputDialog, QTabWidget, QDockWidget
)

from llama_launcher.core.spec import (
    Profile, RouterMember, member_model_id, slugify,
)
from llama_launcher.core.router_preset import render_preset
from llama_launcher.core.pathmap import host_to_container
from llama_launcher.core.validation import (
    LOOPBACK_HOSTS, dial_host,
)
from llama_launcher.store.profiles import (
    default_base_dir, list_profiles, save_profile, delete_profile,
    load_config,
)
# health, router_api, gpu, metrics and model_info are no longer called
# directly in this file (that behavior moved to MonitorController/
# LaunchController/ReportController -- ui/controllers/monitor_controller.py,
# ui/controllers/launch_controller.py, ui/controllers/report_controller.py)
# but stay imported here too: the test suite monkeypatches them as
# `mw.health.probe_health` / `llama_launcher.ui.main_window.router_api.*` /
# `mw.gpu.query_gpus` / `mw.metrics.fetch_metrics` / `mw.model_info.*`, and
# all these names resolve to the same module objects the controllers' own
# imports use, so the patch still reaches the real call sites.
from llama_launcher.services import runtime, terminal, registry, health, metrics, gpu, model_info
from llama_launcher.ui.panels.configure_panel import ConfigurePanel
from llama_launcher.ui.panels.monitor_panel import MonitorPanel
from llama_launcher.ui.panels.benchmark_panel import BenchmarkPanel
from llama_launcher.ui.panels.stats_panel import StatsPanel
from llama_launcher.ui.widgets.router_models_table import RouterModelsTable
from llama_launcher.ui.widgets.status_banner import StatusBanner
from llama_launcher.services import api_key as api_key_store
from llama_launcher.services import router_api


def base_dir():
    return default_base_dir()


# StatsWorker/build_monitor_data used to be defined in this module; they now
# live in monitor_controller.py (moved along with the status/instances/
# monitor/log-follower/stats/router-poll behavior that uses them) but are
# re-exported here too, since the test suite still reaches them as
# `llama_launcher.ui.main_window.StatsWorker` /
# `llama_launcher.ui.main_window.build_monitor_data`.
#
# This import is placed after base_dir() (which monitor_controller.py imports
# back, lazily, per-method -- see its local `from llama_launcher.ui.main_window
# import base_dir` calls) so that name is already bound on this module by the
# time those calls resolve.
from llama_launcher.ui.controllers.monitor_controller import (  # noqa: E402
    MonitorController, StatsWorker, build_monitor_data,
)

# _UpdateWorker used to be defined in this module; it now lives in
# launch_controller.py (moved along with the launch/stop/restart/image-fetch/
# detect/update-check behavior that uses it) but is re-exported here too,
# since the test suite still reaches it as
# `llama_launcher.ui.main_window._UpdateWorker` (both names bind the same
# class object, so a monkeypatch on either resolves for both).
from llama_launcher.ui.controllers.launch_controller import (  # noqa: E402
    LaunchController, _UpdateWorker,
)

# BenchmarkWorker used to be defined in this module; it now lives in
# benchmark_controller.py (moved along with the benchmark run lifecycle
# behavior that uses it). No test reaches it via main_window's namespace, so
# it is not re-exported here -- only BenchmarkController is needed.
from llama_launcher.ui.controllers.benchmark_controller import (  # noqa: E402
    BenchmarkController,
)

# Owns report/export/web-ui behavior (see report_controller.py). No worker
# state moved with it, so unlike the controllers above it needs no re-exported
# helper class -- only ReportController is needed.
from llama_launcher.ui.controllers.report_controller import (  # noqa: E402
    ReportController,
)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Llama Launcher")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Owns status/instances/monitor/log-follower/stats/router-poll behavior
        # (see monitor_controller.py) and the state it reads/writes between
        # ticks. Built here -- before any widget/signal wiring below -- so
        # that state exists from the very start of __init__, ahead of any
        # signal that could synchronously reach it.
        self._monitor = MonitorController(self)
        # Owns launch/stop/restart/enable-metrics + image fetch/detect/update
        # behavior (see launch_controller.py), plus _fetch_worker/_update_worker/
        # _stop_proc/_update_timer. Built here too, ahead of the launch/stop/
        # restart/detect/fetch button .connect() calls below (ConfigurePanel) and
        # _autofill_image_if_empty() further down, both of which reach it.
        self._launch = LaunchController(self)
        # Owns the benchmark run lifecycle (see benchmark_controller.py), plus
        # _benchmark_thread/_benchmark_worker/_benchmark_profile_name. Built
        # here too, ahead of the BenchmarkPanel signal .connect() calls below.
        self._benchmark = BenchmarkController(self)
        # Owns report/export/web-ui behavior (see report_controller.py). Owns
        # no workers -- no drain() needed in _stop_timers(). Built here for
        # consistency with the other controllers, ahead of the report/export/
        # web-ui button .connect() calls below (ConfigurePanel).
        self._report = ReportController(self)
        # ConfigurePanel's own __init__ wires its lifecycle/report/fetch
        # buttons straight to self.window._launch.<m> / self.window._report.<m>
        # (Task 6 repoint off the MainWindow facade), so it must be built AFTER
        # all four controllers above exist as instance attributes.
        self._configure_panel = ConfigurePanel(self)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self._configure_panel.configure_tab, "Configure")
        self.monitor_panel = MonitorPanel()
        self.monitor_panel.enable_metrics_requested.connect(self._launch._on_enable_metrics)
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
        self.benchmark_panel.benchmark_run_requested.connect(self._benchmark._on_benchmark_run)
        self.benchmark_panel.benchmark_cancel_requested.connect(self._benchmark._on_benchmark_cancel)
        self.benchmark_panel.benchmark_clear_requested.connect(self._benchmark._on_benchmark_clear)
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
        root.addWidget(self._configure_panel._config_bottom)
        buttons = QHBoxLayout()
        for b in (self._configure_panel.launch_btn, self._configure_panel.stop_btn,
                  self._configure_panel.restart_btn, self._configure_panel.web_ui_btn):
            buttons.addWidget(b)
        buttons.addWidget(self._configure_panel.detached_check)
        root.addLayout(buttons)

        # profile bar (added to the top of root via insertLayout)
        bar = QHBoxLayout()
        self.stats_toggle_btn = QPushButton("📊 Stats")
        self.stats_toggle_btn.setCheckable(True)
        self.stats_toggle_btn.setToolTip("Show/hide the live stats panel (Ctrl+Shift+S)")
        self.status_label = QLabel("● stopped")
        bar.addWidget(QLabel("Name"))
        bar.addWidget(self._configure_panel.name_edit, 1)
        bar.addWidget(self._configure_panel.profile_combo, 1)
        for b in (self._configure_panel.save_btn, self._configure_panel.save_as_btn,
                  self._configure_panel.delete_btn, self._configure_panel.report_btn,
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
        self._configure_panel._on_mode_changed()

        self._configure_panel.refresh_preview()

        _stats_cfg = load_config(base_dir())
        if _stats_cfg.get("stats_open", False):
            self.stats_toggle_btn.setChecked(True)     # shows the dock
        _w = int(_stats_cfg.get("stats_width", 320) or 320)
        self.resizeDocks([self.stats_dock], [_w], Qt.Horizontal)

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
            menu.addAction("Launch", self._launch.on_launch)
            menu.addAction("Stop", self._launch.on_stop)
            menu.addSeparator()
            menu.addAction("Quit", self.quit_app)
            self.tray.setContextMenu(menu)
            self.tray.show()
        else:
            self.tray = None

        self._configure_panel.lora_panel.set_browse_resolver(
            lambda h: host_to_container(h, self._configure_panel.mounts_panel.mounts())
        )

        # Auto-insert the local image when there's exactly one and none is set yet.
        self._launch._autofill_image_if_empty()

        from PySide6.QtCore import QTimer
        self._status_timer = QTimer(self)
        interval = load_config(base_dir()).get("monitor_interval_ms", 2000)
        self._status_timer.setInterval(interval)
        self._status_timer.timeout.connect(self._monitor.update_status)
        self._status_timer.start()

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

    # -- LaunchController state forwarders -----------------------------------
    # This state now lives on self._launch (see ui/controllers/
    # launch_controller.py); these properties keep it reachable as
    # window.<name> for the test suite (_fetch_worker is set directly by
    # tests exercising _stop_timers's drain path; _fetch_repo is read after
    # on_fetch_latest starts a fetch).
    @property
    def _fetch_worker(self):
        return self._launch._fetch_worker

    @_fetch_worker.setter
    def _fetch_worker(self, value):
        self._launch._fetch_worker = value

    @property
    def _fetch_repo(self):
        return self._launch._fetch_repo

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

    # -- ConfigurePanel behavior delegators ----------------------------------
    # These methods now live on self._configure_panel (Task 2 of the
    # main_window decomposition); MainWindow keeps a one-line forwarder for
    # each so both the test suite and MainWindow's own not-yet-moved code
    # (which calls them as self.<method>()) keep working unchanged.
    def _profile_name(self) -> str:
        return self._configure_panel._profile_name()

    def _container_name(self) -> str:
        return f"llama-{slugify(self._configure_panel._profile_name())}"

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
        if self.tabs.currentWidget() is self._configure_panel.configure_tab:
            self.refresh_router_panel_header()
        # Command preview / api-key / harness only make sense while configuring,
        # so hide the bottom strip on the Monitor/Benchmark tabs.
        self._configure_panel._config_bottom.setVisible(
            self.tabs.currentWidget() is self._configure_panel.configure_tab)

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
        self._configure_panel.profile_combo.clear()
        self._profiles = {p.name: p for p in list_profiles(base_dir())}
        self._configure_panel.profile_combo.addItems(list(self._profiles.keys()))

    def _on_pick_profile(self, _index):
        self._monitor._stop_log_follower()
        name = self._configure_panel.profile_combo.currentText()
        if name in self._profiles:
            self._configure_panel.load_profile(self._profiles[name])

    def save_current_profile(self):
        p = self._configure_panel.current_profile()       # name comes from the Name field
        self._configure_panel._profile = p
        save_profile(p, base_dir())
        self._reload_profile_list()
        self._configure_panel.profile_combo.setCurrentText(p.name)

    def save_as_profile(self):
        name, ok = QInputDialog.getText(self, "Save As", "Profile name:",
                                        text=self._configure_panel._profile_name())
        if ok and name:
            self._configure_panel.name_edit.setText(name)
            self.save_current_profile()

    def delete_current_profile(self):
        name = self._configure_panel.profile_combo.currentText()
        if name:
            delete_profile(name, base_dir())
            self._reload_profile_list()

    def router_base_dir(self):
        return base_dir()

    def router_api_key(self) -> str:
        p = self._configure_panel.current_profile()
        return (api_key_store.resolve_api_key(self.router_base_dir(), p)
                or api_key_store.ensure_api_key(self.router_base_dir(), p.name))

    def prepare_router_files(self) -> tuple:
        """Write models.ini + api-key for the current router. Returns (dir, warnings)."""
        name = self._configure_panel._profile_name()
        result = render_preset(self._configure_panel.member_pairs())
        api_key_store.prepare_launch_key(self.router_base_dir(), self._configure_panel.current_profile())
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
            api_key_store.set_profile_key(base, self._configure_panel._profile_name(), value)
        self.refresh_router_panel_header()
        self._notify_key_change_needs_relaunch()

    def _notify_key_change_needs_relaunch(self) -> None:
        """The running router keeps serving the key it launched with -- a key
        change here only takes effect on the NEXT launch. Without this, a user
        who copies the newly-shown key into their harness while the router is
        still up gets a bare 401 with nothing in the GUI explaining why."""
        p = self._configure_panel.current_profile()
        if runtime.container_state(self._container_name(),
                                   p.runtime.binary) == "running":
            QMessageBox.information(
                self, "Relaunch needed",
                "The router is running. Relaunch it for the new API key to "
                "take effect.")

    def _set_router_connected(self, connected: bool) -> None:
        self._configure_panel.configure_status.set_connected(connected)
        self.monitor_status.set_connected(connected)

    def _set_router_error(self, text: str) -> None:
        self._configure_panel.configure_status.set_error(text)
        self.monitor_status.set_error(text)

    def _set_router_exposure(self, text: str) -> None:
        self._configure_panel.configure_status.set_exposure_warning(text)
        self.monitor_status.set_exposure_warning(text)

    def refresh_router_panel_header(self) -> None:
        p = self._configure_panel.current_profile()
        if p.mode != "router":
            # Clear relocated router state so a previous router's exposure
            # banner, API key, and harness endpoint don't linger on the
            # Configure/Monitor tabs after switching to an unrelated profile.
            self._set_router_exposure("")
            self._configure_panel.api_key_box.set_key("")
            self._configure_panel.harness_box.harness_text.setPlainText("")
            return
        host = p.runtime.bind_host
        display_host = dial_host(host)
        port = p.settings.get("port", 8080)
        # A router without a key is unusable, and the harness block exists so the
        # key can be copied BEFORE the first launch. Generating here is
        # idempotent and is the only side effect on this path.
        key = (api_key_store.resolve_api_key(self.router_base_dir(), p)
               or api_key_store.ensure_api_key(self.router_base_dir(), p.name))
        self._configure_panel.api_key_box.set_key(key)
        self._configure_panel.harness_box.set_endpoint(
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
        return self._launch._report_launch_error(text, show_dialog=show_dialog)

    def adopt_running_containers(self) -> list:
        return self._launch.adopt_running_containers()

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
        return self._launch._validate_or_warn()

    def vram_check(self) -> str | None:
        return self._launch.vram_check()

    def on_launch(self):
        return self._launch.on_launch()

    def _spawn_async(self, argv: list[str], on_done=None, on_error=None):
        return self._launch._spawn_async(argv, on_done=on_done, on_error=on_error)

    def on_stop(self):
        return self._launch.on_stop()

    def on_restart(self):
        return self._launch.on_restart()

    def _on_enable_metrics(self):
        return self._launch._on_enable_metrics()

    def _resolve_benchmark_member(self, p: Profile, model_scope: str | None):
        return self._benchmark._resolve_benchmark_member(p, model_scope)

    def _prepare_benchmark(self, p: Profile):
        return self._benchmark._prepare_benchmark(p)

    def _run_benchmark_sync(self, cfg: dict, run_benchmark=None) -> None:
        return self._benchmark._run_benchmark_sync(cfg, run_benchmark=run_benchmark)

    def _on_benchmark_run(self, cfg: dict) -> None:
        return self._benchmark._on_benchmark_run(cfg)

    def _on_benchmark_thread_done(self) -> None:
        return self._benchmark._on_benchmark_thread_done()

    def _on_benchmark_cancel(self) -> None:
        return self._benchmark._on_benchmark_cancel()

    def _on_benchmark_clear(self) -> None:
        return self._benchmark._on_benchmark_clear()

    def _on_benchmark_finished(self, run) -> None:
        return self._benchmark._on_benchmark_finished(run)

    def _on_benchmark_failed(self, msg: str) -> None:
        return self._benchmark._on_benchmark_failed(msg)

    @property
    def _benchmark_thread(self):
        return self._benchmark._benchmark_thread

    def on_fetch_latest(self):
        return self._launch.on_fetch_latest()

    def _on_fetch_found(self, tag: str) -> None:
        return self._launch._on_fetch_found(tag)

    def _on_fetch_failed(self, msg: str) -> None:
        return self._launch._on_fetch_failed(msg)

    def _on_fetch_finished(self) -> None:
        return self._launch._on_fetch_finished()

    def detect_image(self):
        return self._launch.detect_image()

    def _autofill_image_if_empty(self):
        return self._launch._autofill_image_if_empty()

    def check_for_update(self, tags: list[str]) -> str | None:
        return self._launch.check_for_update(tags)

    def run_update_check(self):
        return self._launch.run_update_check()

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
        """Stop background timers/workers so a torn-down window can't keep
        firing update_status, keep polling stats, or leave a QThread running
        past the window's lifetime. Idempotent — QTimer.stop() on a stopped
        timer is a no-op, and each drain() below is itself a no-op when its
        controller owns nothing currently running.

        self._status_timer (update_status polling) is the only timer left on
        MainWindow, stopped directly here. The update-check singleShot timer
        lives on LaunchController but is also stopped directly here, early --
        ahead of MonitorController.drain(), which pumps the event loop and
        could otherwise let it fire mid-teardown. Everything else -- the
        stats worker and in-flight monitor gather (MonitorController.
        drain()), the fetch/update QThreads (LaunchController.drain()), and a
        running benchmark QThread (BenchmarkController.drain()) -- is owned
        by its controller, so teardown is otherwise just a sequence of
        drain() calls in controller-construction order.
        """
        self._status_timer.stop()
        # Stop the update-check singleShot timer early, before drain() below
        # pumps the event loop (MonitorController.drain()'s stats-worker wait
        # loop calls QCoreApplication.processEvents()) -- otherwise the timer
        # can fire mid-teardown and spawn an _UpdateWorker. LaunchController.
        # drain() also stops it (idempotent), but this restores the original
        # early-stop ordering.
        _update_timer = getattr(self._launch, "_update_timer", None)
        if _update_timer is not None:
            _update_timer.stop()
        self._monitor.drain()
        self._launch.drain()
        self._benchmark.drain()

    def _on_export_sh(self):
        return self._report._on_export_sh()

    def open_web_ui(self):
        return self._report.open_web_ui()

    def export_sh(self, path: str):
        return self._report.export_sh(path)

    def gather_report_data(self) -> dict:
        return self._report.gather_report_data()

    def _metrics_report_text(self, p: Profile) -> str:
        return self._report._metrics_report_text(p)

    def _save_report(self, md: str, ts: str | None = None) -> Path:
        return self._report._save_report(md, ts)

    def on_generate_report(self):
        return self._report.on_generate_report()

    def closeEvent(self, event):
        if getattr(self, "_really_quit", False) or not self._minimize_to_tray:
            self._monitor._stop_log_follower()
            self._stop_timers()
            event.accept()
            from PySide6.QtWidgets import QApplication
            QApplication.instance().quit()
        else:
            event.ignore()
            self.hide()

    def quit_app(self):
        self._really_quit = True
        self._monitor._stop_log_follower()
        self._stop_timers()
        from PySide6.QtWidgets import QApplication
        QApplication.instance().quit()

    def apply_model_caps(self) -> None:
        return self._configure_panel.apply_model_caps()

    def _suggestion_index(self, caps):
        return self._configure_panel._suggestion_index(caps)

    @staticmethod
    def _dot_state_for(key, described, sugg_by_key, reason_by_key):
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
        return ConfigurePanel._meta_caps_text(meta, size, caps)
