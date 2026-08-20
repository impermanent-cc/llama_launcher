import pytest

import llama_launcher.ui.main_window as mw
from llama_launcher.core.spec import Mount, Profile, RouterMember
from llama_launcher.services import benchmark, benchmark_store
from llama_launcher.services.benchmark import BenchmarkRow, BenchmarkRun
from llama_launcher.store import profiles as store
from llama_launcher.store.profiles import default_base_dir
from llama_launcher.ui.controllers.monitor_controller import MonitorController
from llama_launcher.ui.main_window import MainWindow


@pytest.fixture
def win(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _canned_run(timestamp="2026-08-07T00:00:00"):
    return BenchmarkRun(
        timestamp=timestamp,
        sizes=[128, 512],
        n_predict=64,
        warmup=1,
        repeats=3,
        rows=[
            BenchmarkRow(target_size=128, prompt_n=128, pp_tok_s=1000.0,
                        gen_tok_s=50.0, total_s=1.0),
            BenchmarkRow(target_size=512, prompt_n=512, pp_tok_s=900.0,
                        gen_tok_s=45.0, total_s=2.0),
        ],
        snapshot={"model": "a.gguf", "image": "img", "mode": "server"},
    )


def _load_server_profile(win):
    p = Profile(name="Solo", image="img", model="/models/a.gguf",
               mounts=[Mount(host="/h", container="/models")],
               settings={"port": 8080})
    win._configure_panel.load_profile(p)
    return p


def _fake_run_benchmark(calls=None):
    def _run(client, sizes, n_predict, warmup, repeats, snapshot, timestamp,
             should_cancel=None):
        if calls is not None:
            calls.append((sizes, n_predict, warmup, repeats))
        return _canned_run(timestamp)
    return _run


def test_run_benchmark_sync_saves_one_run_and_updates_the_table(win, monkeypatch):
    p = _load_server_profile(win)
    calls = []
    monkeypatch.setattr(benchmark, "run_benchmark", _fake_run_benchmark(calls))

    win._benchmark._run_benchmark_sync({"sizes": [128, 512], "n_predict": 64, "warmup": 1, "repeats": 3})

    runs = benchmark_store.load(default_base_dir(), p.name)
    assert len(runs) == 1
    # grouped: one model header row + the two metric rows
    assert win.benchmark_panel.bench_table.rowCount() == 3
    assert "a.gguf" in win.benchmark_panel.bench_table.item(0, 0).text()
    # the sizes/n_predict/warmup/repeats from cfg reached run_benchmark unchanged
    assert calls == [([128, 512], 64, 1, 3)]


def test_clear_button_clears_store_and_table(win, monkeypatch):
    p = _load_server_profile(win)
    monkeypatch.setattr(benchmark, "run_benchmark", _fake_run_benchmark())
    win._benchmark._run_benchmark_sync({"sizes": [128, 512], "n_predict": 64, "warmup": 1, "repeats": 3})
    assert benchmark_store.load(default_base_dir(), p.name)          # something saved

    win.benchmark_panel.benchmark_clear_requested.emit()

    assert benchmark_store.load(default_base_dir(), p.name) == []    # on-disk wiped
    assert win.benchmark_panel.bench_table.rowCount() == 0           # view wiped


def test_second_run_grows_history_and_shows_a_delta(win, monkeypatch):
    p = _load_server_profile(win)
    monkeypatch.setattr(benchmark, "run_benchmark", _fake_run_benchmark())

    deltas = []
    orig_show = win.benchmark_panel.show_benchmark_run

    def spy(run, delta):
        deltas.append(delta)
        orig_show(run, delta)

    monkeypatch.setattr(win.benchmark_panel, "show_benchmark_run", spy)

    cfg = {"sizes": [128, 512], "n_predict": 64, "warmup": 1, "repeats": 3}
    win._benchmark._run_benchmark_sync(cfg)
    win._benchmark._run_benchmark_sync(cfg)

    runs = benchmark_store.load(default_base_dir(), p.name)
    assert len(runs) == 2
    assert deltas[0] is None        # first run: nothing stored yet to diff against
    assert deltas[1] is not None    # second run: delta vs the first
    # two grouped runs, each a header row + two metric rows
    assert win.benchmark_panel.bench_table.rowCount() == 6


def test_run_benchmark_sync_refuses_when_router_has_no_loaded_model(win, monkeypatch):
    p = Profile(name="Host", mode="router", image="img",
               members=[RouterMember(profile="Qwen")], settings={"port": 8080})
    win._configure_panel.load_profile(p)
    win._monitor._router_statuses = {}       # nothing loaded -> nothing to benchmark

    called = []
    monkeypatch.setattr(benchmark, "run_benchmark", _fake_run_benchmark(called))

    win._benchmark._run_benchmark_sync({"sizes": [128], "n_predict": 64, "warmup": 1, "repeats": 1})

    assert called == []
    assert benchmark_store.load(default_base_dir(), p.name) == []


def test_on_benchmark_failed_saves_nothing(win, monkeypatch):
    p = _load_server_profile(win)

    def raising(*_a, **_k):
        raise benchmark.BenchmarkError("boom")

    monkeypatch.setattr(benchmark, "run_benchmark", raising)

    win._benchmark._run_benchmark_sync({"sizes": [128], "n_predict": 64, "warmup": 1, "repeats": 1})

    assert benchmark_store.load(default_base_dir(), p.name) == []
    assert "boom" in win.benchmark_panel.bench_progress.text()


def test_stop_timers_cancels_and_stops_a_running_benchmark_thread(win, monkeypatch):
    _load_server_profile(win)
    import time

    def slow_run(client, sizes, n_predict, warmup, repeats, snapshot, timestamp,
                should_cancel=None):
        # Loops until cancelled (or ~2s worst case) so _stop_timers's cancel
        # call is what actually ends this, not the work finishing on its own.
        for _ in range(200):
            if should_cancel is not None and should_cancel():
                raise benchmark.BenchmarkError("cancelled")
            time.sleep(0.01)
        return _canned_run(timestamp)

    monkeypatch.setattr(benchmark, "run_benchmark", slow_run)

    win._benchmark._on_benchmark_run({"sizes": [128], "n_predict": 64, "warmup": 0, "repeats": 1})
    thread = win._benchmark._benchmark_thread
    assert thread is not None

    win._stop_timers()

    assert thread.isFinished()
    assert win._benchmark._benchmark_thread is None


# --- Task 6: availability gate + per-profile history on reset ---------------

def _ready_status_stubs(monkeypatch):
    """Stub the status-poll dependencies so update_status() reads server/ready
    without touching the network or real container runtime."""
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    monkeypatch.setattr(mw.runtime, "container_state", lambda name, binary, connection="": "running")
    monkeypatch.setattr(mw.health, "probe_health", lambda port, **kw: "ready")
    monkeypatch.setattr(mw.metrics, "fetch_props", lambda *a, **k: None)
    monkeypatch.setattr(MonitorController, "_log_follower_active", lambda self: True)
    monkeypatch.setattr(MonitorController, "_update_spec_stats", lambda self, p: None)


def _member_profile(base, name="Qwen"):
    p = Profile(name=name, image="img", model="/models/qwen.gguf",
               mounts=[Mount(host="/mnt/models", container="/models")],
               settings={"ctx-size": 8192})
    store.save_profile(p, base)
    return p


def test_benchmark_run_enabled_when_server_running_and_ready(win, monkeypatch):
    _load_server_profile(win)
    _ready_status_stubs(monkeypatch)

    win._monitor.update_status()

    assert win.benchmark_panel.bench_run_btn.isEnabled()


def test_benchmark_run_disabled_when_not_running(win, monkeypatch):
    _load_server_profile(win)
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    monkeypatch.setattr(mw.runtime, "container_state", lambda name, binary, connection="": "exited")
    # Prove the gate actively disables rather than merely defaulting to off.
    win.benchmark_panel.set_benchmark_available(True)

    win._monitor.update_status()

    assert not win.benchmark_panel.bench_run_btn.isEnabled()


def test_benchmark_run_disabled_when_binary_unavailable(win, monkeypatch):
    _load_server_profile(win)
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: False)
    win.benchmark_panel.set_benchmark_available(True)

    win._monitor.update_status()

    assert not win.benchmark_panel.bench_run_btn.isEnabled()


def test_benchmark_run_disabled_for_router_with_no_pollable_model(win, monkeypatch):
    p = Profile(name="Host", mode="router", image="img",
               members=[RouterMember(profile="Qwen")], settings={"port": 8080})
    win._configure_panel.load_profile(p)
    _ready_status_stubs(monkeypatch)
    monkeypatch.setattr(MonitorController, "refresh_router_models", lambda self: None)
    win._monitor._router_statuses = {}       # nothing loaded -> nothing pollable

    win._monitor.update_status()

    assert not win.benchmark_panel.bench_run_btn.isEnabled()


def test_benchmark_run_enabled_for_router_with_pollable_model(win, monkeypatch):
    p = Profile(name="Host", mode="router", image="img",
               members=[RouterMember(profile="Qwen")], settings={"port": 8080})
    win._configure_panel.load_profile(p)
    _ready_status_stubs(monkeypatch)
    monkeypatch.setattr(MonitorController, "refresh_router_models", lambda self: None)
    win._monitor._router_statuses = {"qwen": "loaded"}

    win._monitor.update_status()

    assert win.benchmark_panel.bench_run_btn.isEnabled()


def test_history_loaded_on_server_launch_reset(win, monkeypatch):
    p = _load_server_profile(win)
    benchmark_store.append(default_base_dir(), p.name, _canned_run())
    monkeypatch.setattr(mw.terminal, "launch", lambda *a, **k: None)
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)

    win._launch.on_launch()

    assert win.benchmark_panel.bench_table.rowCount() >= 1
    assert "a.gguf" in win.benchmark_panel.bench_table.item(0, 0).text()


def test_history_loaded_on_router_launch_reset(win, monkeypatch):
    base = store.default_base_dir()
    _member_profile(base)
    p = Profile(name="Host", mode="router", image="img",
               mounts=[Mount(host="/mnt/models", container="/models")],
               members=[RouterMember(profile="Qwen")], settings={"port": 8080})
    win._configure_panel.load_profile(p)
    benchmark_store.append(base, p.name, _canned_run())
    monkeypatch.setattr(win._launch, "_validate_or_warn", lambda: True)
    monkeypatch.setattr(win._launch, "vram_check", lambda: None)
    monkeypatch.setattr(win._launch, "_spawn_async", lambda argv, on_done=None, on_error=None: None)
    monkeypatch.setattr(mw.runtime, "container_state", lambda name, binary, connection="": "exited")

    win._launch.on_launch()

    assert win.benchmark_panel.bench_table.rowCount() >= 1
