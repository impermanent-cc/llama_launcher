import llama_launcher.ui.main_window as mw
from llama_launcher.core.spec import Profile, Mount, Runtime
from llama_launcher.core.props import PropsInfo


def _profile():
    return Profile(name="m", image="img", runtime=Runtime(binary="podman"),
                   mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
                   model="/models/m.gguf", settings={"port": 8080, "metrics": True})


def test_on_stop_clears_log_follower_and_spawns_async_stop(qtbot, monkeypatch):
    """on_stop() kills the log follower immediately and spawns `podman stop`
    asynchronously (never blocking the UI thread) with the right argv."""
    spawned = {}
    monkeypatch.setattr(mw.MainWindow, "_spawn_async",
                        lambda self, argv, on_done=None: spawned.setdefault("argv", argv))
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w.load_profile(_profile())   # profile name "m" -> container llama-m

    killed = []

    class _FakeProc:
        def kill(self):
            killed.append(True)

    w._log_proc = _FakeProc()
    w.on_stop()
    assert w._log_proc is None
    assert killed == [True]
    assert spawned["argv"] == ["podman", "stop", "-t", "10", "llama-m"]


def test_collect_monitor_data(qtbot, monkeypatch):
    monkeypatch.setattr(mw.metrics, "fetch_metrics",
                        lambda port, timeout=1.0, **kw: {"llamacpp:predicted_tokens_seconds": 50.0,
                                                         "llamacpp:prompt_tokens_seconds": 200.0})
    monkeypatch.setattr(mw.metrics, "fetch_slots", lambda port, timeout=1.0, **kw:
                        [{"n_ctx": 100, "n_prompt_tokens_processed": 40}])
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda: [])
    monkeypatch.setattr(mw.runtime, "stats", lambda name, b: {"cpu_perc": "9%", "mem_usage": "1G / 16G"})
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w.load_profile(_profile())
    d = w.collect_monitor_data()
    assert d["tok_s"] == 50.0 and d["prompt_tok_s"] == 200.0
    assert abs(d["kv_pct"] - 0.40) < 1e-9
    assert d["metrics_on"] is True
    assert d["cpu"] == "9%"


def test_collect_monitor_data_reports_speculating(qtbot, monkeypatch):
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    monkeypatch.setattr(mw.metrics, "fetch_slots",
                        lambda *a, **k: [{"speculative": True, "n_ctx": 4096}])
    monkeypatch.setattr(mw.metrics, "fetch_metrics", lambda *a, **k: {})
    monkeypatch.setattr(mw.runtime, "stats", lambda name, binary: {})
    monkeypatch.setattr(mw.runtime, "started_at", lambda name, binary: None)
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda: [])
    w = mw.MainWindow()
    qtbot.addWidget(w)
    assert w.collect_monitor_data()["speculating"] is True


def _ready(monkeypatch):
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    monkeypatch.setattr(mw.runtime, "container_state", lambda name, binary: "running")
    monkeypatch.setattr(mw.health, "probe_health", lambda port, **kw: "ready")
    monkeypatch.setattr(mw.MainWindow, "collect_monitor_data", lambda self: {})
    monkeypatch.setattr(mw.MainWindow, "_log_follower_active", lambda self: True)
    monkeypatch.setattr(mw.MainWindow, "_update_spec_stats", lambda self, p: None)


def test_props_fetched_once_when_ready(qtbot, monkeypatch):
    _ready(monkeypatch)
    calls = []
    monkeypatch.setattr(mw.metrics, "fetch_props",
                        lambda *a, **k: calls.append(1) or PropsInfo("b", 1, "a", 1, {}))
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w.update_status()
    w.update_status()
    assert len(calls) == 1                 # cached after first success


def test_props_not_fetched_while_loading(qtbot, monkeypatch):
    _ready(monkeypatch)
    monkeypatch.setattr(mw.health, "probe_health", lambda port, **kw: "loading")
    calls = []
    monkeypatch.setattr(mw.metrics, "fetch_props", lambda *a, **k: calls.append(1))
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w.update_status()
    assert calls == []                     # only fetch once the model is ready


def test_failed_props_fetch_retries_next_poll(qtbot, monkeypatch):
    _ready(monkeypatch)
    calls = []
    monkeypatch.setattr(mw.metrics, "fetch_props",
                        lambda *a, **k: calls.append(1) or None)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w.update_status()
    w.update_status()
    assert len(calls) == 2                 # None result is not cached; retries


def test_router_model_switch_refetches(qtbot, monkeypatch):
    # NOTE: adapted from the brief's two mechanisms, both to de-brittle this
    # test against MainWindow's background _status_timer (2000ms), which
    # isn't guaranteed stopped when an earlier test's window is torn down --
    # a stray timeout can invoke update_status() an extra, unpredictable
    # number of times (on this OR another still-alive window) while this
    # test runs:
    #  1. A mutable "current id" dict replaces the brief's strict
    #     `iter([...])`, since a one-shot iterator raises StopIteration on
    #     any unplanned extra call, from any window sharing this class-level
    #     patch.
    #  2. Assertions inspect `w`'s own `_props`/`_props_model` cache fields
    #     directly instead of a shared global call counter, since a stray
    #     call landing on a *different* leftover window would independently
    #     populate that window's own cache and inflate a shared counter
    #     without this test's own `w` being involved at all.
    # The cache-key behavior under test -- re-fetch on router model-id
    # change, cache-hit while unchanged -- is unaffected by either change.
    _ready(monkeypatch)
    monkeypatch.setattr(mw.MainWindow, "current_profile",
                        lambda self: _router_profile())
    monkeypatch.setattr(mw.MainWindow, "_router_host", lambda self, p: "127.0.0.1")
    current = {"id": "model-a"}
    monkeypatch.setattr(mw.MainWindow, "_router_pollable_model",
                        lambda self: current["id"])
    monkeypatch.setattr(mw.metrics, "fetch_props",
                        lambda *a, **k: PropsInfo("b", 1, "a", 1, {}))
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w.update_status()   # model-a -> fetch
    assert w._props_model == "model-a"
    first = w._props
    assert first is not None
    w.update_status()   # model-a -> cached (unchanged instance)
    assert w._props is first
    current["id"] = "model-b"
    w.update_status()   # model-b -> re-fetch
    assert w._props_model == "model-b"
    assert w._props is not first


def _router_profile():
    from llama_launcher.core.spec import Profile, Runtime
    return Profile(name="r", image="img", mode="router", runtime=Runtime())
