"""Composition assertions for the MainWindow decomposition.

These tests pin structure, not behavior: ConfigurePanel and the four
controllers exist as separate objects on MainWindow (`w._configure_panel`,
`w._monitor`, `w._launch`, `w._benchmark`, `w._report`), the LaunchController
owns _spawn_async, and MainWindow's own teardown orchestration reaches all
three worker-owning controllers.
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
    """_spawn_async is a real LaunchController method, not a forward: the autouse
    fixture class-patches it there ahead of every UI test."""
    w = MainWindow()
    qtbot.addWidget(w)
    assert hasattr(w._launch, "_spawn_async")
    assert type(w._launch)._spawn_async is LaunchController._spawn_async


def test_stop_timers_drains_all_three_worker_owning_controllers(qtbot, monkeypatch):
    """_stop_timers stays on MainWindow itself (it coordinates draining across
    controllers) and reaches all three drain()-owning controllers.
    """
    w = MainWindow()
    qtbot.addWidget(w)
    drained = []
    monkeypatch.setattr(w._monitor, "drain", lambda: drained.append("monitor"))
    monkeypatch.setattr(w._launch, "drain", lambda: drained.append("launch"))
    monkeypatch.setattr(w._benchmark, "drain", lambda: drained.append("benchmark"))
    w._stop_timers()
    assert set(drained) == {"monitor", "launch", "benchmark"}
