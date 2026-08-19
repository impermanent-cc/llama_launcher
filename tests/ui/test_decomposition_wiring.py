"""Additive composition assertions for the MainWindow decomposition (Tasks 1-8).

These tests don't exercise new behavior -- every path here is already covered
by the pre-existing 854 tests. This file exists purely to pin the *structure*
the decomposition promised: that ConfigurePanel + the four controllers
actually exist as separate objects on MainWindow, that the LaunchController
owns _spawn_async, and that MainWindow's own teardown orchestration still
reaches all three worker-owning controllers.

The forwarding facade (`@property` widget-forwards + one-line method
delegators on MainWindow, e.g. `w.image_edit`, `w.on_launch()`) is being
deleted (facade-shrink Task 7), so this file no longer asserts identity
through it -- only the surviving composition (`w._configure_panel`,
`w._monitor`, `w._launch`, `w._benchmark`, `w._report`) is pinned here.
"""
from llama_launcher.ui.main_window import MainWindow
from llama_launcher.ui.panels.configure_panel import ConfigurePanel
from llama_launcher.ui.controllers.monitor_controller import MonitorController
from llama_launcher.ui.controllers.launch_controller import LaunchController
from llama_launcher.ui.controllers.benchmark_controller import BenchmarkController
from llama_launcher.ui.controllers.report_controller import ReportController


def test_configure_panel_and_controllers_exist(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    assert w._configure_panel is not None
    assert isinstance(w._configure_panel, ConfigurePanel)
    assert w._monitor is not None
    assert isinstance(w._monitor, MonitorController)
    assert w._launch is not None
    assert isinstance(w._launch, LaunchController)
    assert w._benchmark is not None
    assert isinstance(w._benchmark, BenchmarkController)
    assert w._report is not None
    assert isinstance(w._report, ReportController)


def test_controllers_hold_window_back_reference(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    assert w._monitor.window is w
    assert w._launch.window is w
    assert w._benchmark.window is w
    assert w._report.window is w


def test_spawn_async_is_owned_by_launch_controller(qtbot):
    """_spawn_async is the highest-risk member of the facade-shrink plan (it's
    class-patched on MainWindow by an autouse fixture ahead of every UI test).
    Pin that it's a real LaunchController method, not just forwarded through."""
    w = MainWindow()
    qtbot.addWidget(w)
    assert hasattr(w._launch, "_spawn_async")
    assert type(w._launch)._spawn_async is LaunchController._spawn_async


def test_stop_timers_drains_all_three_worker_owning_controllers(qtbot, monkeypatch):
    """_stop_timers is the one piece of orchestration Task 7 confirmed stays on
    MainWindow itself (it coordinates draining across controllers); pin that it
    still reaches all three drain()-owning controllers.
    """
    w = MainWindow()
    qtbot.addWidget(w)
    drained = []
    monkeypatch.setattr(w._monitor, "drain", lambda: drained.append("monitor"))
    monkeypatch.setattr(w._launch, "drain", lambda: drained.append("launch"))
    monkeypatch.setattr(w._benchmark, "drain", lambda: drained.append("benchmark"))
    w._stop_timers()
    assert set(drained) == {"monitor", "launch", "benchmark"}
