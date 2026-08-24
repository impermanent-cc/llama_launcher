# subprocess is no longer called directly in this file (open_web_ui's
# subprocess.Popen call moved to ReportController,
# ui/controllers/report_controller.py) but stays imported here too: the test
# suite monkeypatches it as `mw.subprocess.Popen`, and both names resolve to
# the same module object report_controller.py's own import uses, so the patch
# still reaches the real call site.
import subprocess

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QScrollArea, QLabel, QPushButton,
    QMessageBox, QInputDialog, QTabWidget, QDockWidget
)

from llama_launcher.core.spec import (
    member_model_id, slugify,
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


# build_monitor_data used to be defined in this module; it now lives in
# monitor_controller.py (moved along with the status/instances/monitor/
# log-follower/stats/router-poll behavior that uses it) but is re-exported
# here too, since the test suite still reaches it as
# `llama_launcher.ui.main_window.build_monitor_data`.
#
# This import is placed after base_dir() (which monitor_controller.py imports
# back, lazily, per-method -- see its local `from llama_launcher.ui.main_window
# import base_dir` calls) so that name is already bound on this module by the
# time those calls resolve.
from llama_launcher.ui.controllers.monitor_controller import (  # noqa: E402
    MonitorController, build_monitor_data,
)

from llama_launcher.ui.controllers.launch_controller import (  # noqa: E402
    LaunchController,
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
        self.nodes_btn = QPushButton("Nodes…")
        self.nodes_btn.setToolTip("Add/test/remove remote podman-over-SSH nodes")
        self.nodes_btn.clicked.connect(self.open_nodes_dialog)
        self.status_label = QLabel("● stopped")
        bar.addWidget(QLabel("Name"))
        bar.addWidget(self._configure_panel.name_edit, 1)
        bar.addWidget(self._configure_panel.profile_combo, 1)
        for b in (self._configure_panel.save_btn, self._configure_panel.save_as_btn,
                  self._configure_panel.delete_btn, self._configure_panel.report_btn,
                  self.nodes_btn, self.stats_toggle_btn):
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
            if tray_icon.isNull():   # no asset/theme icon found; keep a visible fallback
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

    def _container_name(self) -> str:
        return f"llama-{slugify(self._configure_panel._profile_name())}"

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

    def base_dir(self):
        """Config-root accessor for panels that need it directly (e.g. the
        Configure panel's node list, the Nodes… dialog) -- same underlying
        directory as router_base_dir(), just named for the general case."""
        return base_dir()

    def open_nodes_dialog(self) -> None:
        from llama_launcher.ui.dialogs.nodes_dialog import NodesDialog
        NodesDialog(self.base_dir(), self).exec()
        self._configure_panel.reload_nodes()

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

    def _stop_timers(self) -> None:
        """Stop background timers/workers so a torn-down window can't keep
        firing update_status, keep polling stats, or leave a QThread running
        past the window's lifetime. Idempotent; QTimer.stop() on a stopped
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
