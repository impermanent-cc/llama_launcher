"""Additive wiring assertions for the MainWindow decomposition (Tasks 1-8).

These tests don't exercise new behavior -- every path here is already covered
by the pre-existing 844 tests via the `w.<attr>`/`w.<method>()` facade. This
file exists purely to pin the *structure* the decomposition promised: that
ConfigurePanel + the four controllers actually exist as separate objects on
MainWindow, that widgets forwarded via `@property` really resolve to the
panel's own widgets (not a stale copy), and that a sample of the one-line
behavior delegators actually route to the controller/panel that owns the
behavior, rather than reimplementing it inline on MainWindow.
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
    assert isinstance(w._configure_panel, ConfigurePanel)
    assert isinstance(w._monitor, MonitorController)
    assert isinstance(w._launch, LaunchController)
    assert isinstance(w._benchmark, BenchmarkController)
    assert isinstance(w._report, ReportController)


def test_widget_properties_forward_to_configure_panel(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    # A sample of forwarded widgets -- identity, not just equality, confirms
    # MainWindow holds no separate copy of its own.
    assert w.image_edit is w._configure_panel.image_edit
    assert w.name_edit is w._configure_panel.name_edit
    assert w.model_edit is w._configure_panel.model_edit
    assert w.mode_combo is w._configure_panel.mode_combo
    assert w.engine_combo is w._configure_panel.engine_combo
    assert w.gpu_combo is w._configure_panel.gpu_combo
    assert w.mounts_panel is w._configure_panel.mounts_panel
    assert w.lora_panel is w._configure_panel.lora_panel


def test_controllers_hold_window_back_reference(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    assert w._monitor.window is w
    assert w._launch.window is w
    assert w._benchmark.window is w
    assert w._report.window is w


def test_on_launch_delegates_to_launch_controller(qtbot, monkeypatch):
    w = MainWindow()
    qtbot.addWidget(w)
    called = []
    monkeypatch.setattr(w._launch, "on_launch", lambda: called.append(True))
    w.on_launch()
    assert called == [True]


def test_on_stop_delegates_to_launch_controller(qtbot, monkeypatch):
    w = MainWindow()
    qtbot.addWidget(w)
    called = []
    monkeypatch.setattr(w._launch, "on_stop", lambda: called.append(True))
    w.on_stop()
    assert called == [True]


def test_detect_image_delegates_to_launch_controller(qtbot, monkeypatch):
    w = MainWindow()
    qtbot.addWidget(w)
    called = []
    monkeypatch.setattr(w._launch, "detect_image", lambda: called.append(True))
    w.detect_image()
    assert called == [True]


def test_update_status_delegates_to_monitor_controller(qtbot, monkeypatch):
    w = MainWindow()
    qtbot.addWidget(w)
    called = []
    monkeypatch.setattr(w._monitor, "update_status", lambda: called.append(True))
    w.update_status()
    assert called == [True]


def test_on_benchmark_run_delegates_to_benchmark_controller(qtbot, monkeypatch):
    w = MainWindow()
    qtbot.addWidget(w)
    called = []
    monkeypatch.setattr(w._benchmark, "_on_benchmark_run", lambda cfg: called.append(cfg))
    w._on_benchmark_run({"n": 1})
    assert called == [{"n": 1}]


def test_on_generate_report_delegates_to_report_controller(qtbot, monkeypatch):
    w = MainWindow()
    qtbot.addWidget(w)
    called = []
    monkeypatch.setattr(w._report, "on_generate_report", lambda: called.append(True))
    w.on_generate_report()
    assert called == [True]


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
