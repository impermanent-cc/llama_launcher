import pytest

from llama_launcher.core.spec import Mount, Profile, RouterMember
from llama_launcher.services import benchmark, benchmark_store
from llama_launcher.services.benchmark import BenchmarkRow, BenchmarkRun
from llama_launcher.store.profiles import default_base_dir
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
    win.load_profile(p)
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

    win._run_benchmark_sync({"sizes": [128, 512], "n_predict": 64, "warmup": 1, "repeats": 3})

    runs = benchmark_store.load(default_base_dir(), p.name)
    assert len(runs) == 1
    assert win.monitor_panel.bench_table.rowCount() == 2
    # the sizes/n_predict/warmup/repeats from cfg reached run_benchmark unchanged
    assert calls == [([128, 512], 64, 1, 3)]


def test_second_run_grows_history_and_shows_a_delta(win, monkeypatch):
    p = _load_server_profile(win)
    monkeypatch.setattr(benchmark, "run_benchmark", _fake_run_benchmark())

    deltas = []
    orig_show = win.monitor_panel.show_benchmark_run

    def spy(run, delta):
        deltas.append(delta)
        orig_show(run, delta)

    monkeypatch.setattr(win.monitor_panel, "show_benchmark_run", spy)

    cfg = {"sizes": [128, 512], "n_predict": 64, "warmup": 1, "repeats": 3}
    win._run_benchmark_sync(cfg)
    win._run_benchmark_sync(cfg)

    runs = benchmark_store.load(default_base_dir(), p.name)
    assert len(runs) == 2
    assert deltas[0] is None        # first run: nothing stored yet to diff against
    assert deltas[1] is not None    # second run: delta vs the first
    assert win.monitor_panel.bench_table.rowCount() == 2


def test_run_benchmark_sync_refuses_when_router_has_no_loaded_model(win, monkeypatch):
    p = Profile(name="Host", mode="router", image="img",
               members=[RouterMember(profile="Qwen")], settings={"port": 8080})
    win.load_profile(p)
    win._router_statuses = {}       # nothing loaded -> nothing to benchmark

    called = []
    monkeypatch.setattr(benchmark, "run_benchmark", _fake_run_benchmark(called))

    win._run_benchmark_sync({"sizes": [128], "n_predict": 64, "warmup": 1, "repeats": 1})

    assert called == []
    assert benchmark_store.load(default_base_dir(), p.name) == []


def test_on_benchmark_failed_saves_nothing(win, monkeypatch):
    p = _load_server_profile(win)

    def raising(*_a, **_k):
        raise benchmark.BenchmarkError("boom")

    monkeypatch.setattr(benchmark, "run_benchmark", raising)

    win._run_benchmark_sync({"sizes": [128], "n_predict": 64, "warmup": 1, "repeats": 1})

    assert benchmark_store.load(default_base_dir(), p.name) == []
    assert "boom" in win.monitor_panel.bench_progress.text()


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

    win._on_benchmark_run({"sizes": [128], "n_predict": 64, "warmup": 0, "repeats": 1})
    thread = win._benchmark_thread
    assert thread is not None

    win._stop_timers()

    assert thread.isFinished()
    assert win._benchmark_thread is None
