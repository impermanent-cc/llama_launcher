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


def test_build_monitor_data_gathers_from_a_plain_target(monkeypatch):
    """build_monitor_data() is a pure function of a primitives-only target so it
    can run off the UI thread: it must gather from those primitives, never from
    live widget/profile state."""
    monkeypatch.setattr(mw.metrics, "fetch_metrics",
                        lambda *a, **k: {"llamacpp:predicted_tokens_seconds": 50.0})
    monkeypatch.setattr(mw.metrics, "fetch_slots",
                        lambda *a, **k: [{"n_ctx": 100, "n_prompt_tokens_processed": 40}])
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda: [])
    monkeypatch.setattr(mw.runtime, "stats",
                        lambda name, b: {"cpu_perc": "9%", "mem_usage": "1G / 16G"})
    monkeypatch.setattr(mw.runtime, "started_at", lambda name, b: None)
    target = {"running": True, "port": 8080, "metrics_on": True, "host": "127.0.0.1",
              "key": None, "model_scope": None, "poll": True,
              "name": "llama-x", "binary": "podman"}
    d = mw.build_monitor_data(target)
    assert d["tok_s"] == 50.0 and d["cpu"] == "9%"
    assert abs(d["kv_pct"] - 0.40) < 1e-9


def test_build_monitor_data_returns_none_when_not_running(monkeypatch):
    """A not-running target does no I/O and returns None, so the worker emits
    nothing (an idle/stopped server isn't polled off-thread every second)."""
    called = []
    monkeypatch.setattr(mw.runtime, "stats", lambda *a, **k: called.append(1))
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda: called.append(1))
    assert mw.build_monitor_data({"running": False}) is None
    assert called == []


def test_update_status_populates_monitor_target_off_ui_thread(qtbot, monkeypatch):
    """update_status snapshots the poll inputs into _monitor_target on the UI
    thread; the worker reads that plain dict and does the blocking gather. When
    the container is running the target is marked running=True."""
    _ready(monkeypatch)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w.update_status()
    assert w._monitor_target.get("running") is True


def test_update_status_marks_target_not_running_when_stopped(qtbot, monkeypatch):
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    monkeypatch.setattr(mw.runtime, "container_state", lambda name, binary: "stopped")
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w.update_status()
    assert w._monitor_target.get("running") is False


def test_monitor_summary_gathered_off_thread_is_rendered(qtbot, monkeypatch):
    """End-to-end: the pooled gather's result is rendered on a later tick.

    Tick 1 dispatches the off-thread gather (nothing to render yet); once it
    completes, tick 2 renders the stored result via update_stats. Guards against
    the result being cleared before it can be shown.
    """
    _ready(monkeypatch)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    got = []
    w.monitor_panel.update_stats = lambda d: got.append(d)
    w.update_status()                       # dispatch; _monitor_result still None
    from PySide6.QtCore import QThreadPool
    QThreadPool.globalInstance().waitForDone(2000)
    w.update_status()                       # renders the gathered result
    assert got, "no monitor summary was rendered"
    assert "gpus" in got[-1] and "uptime" in got[-1]


def test_log_updates_are_coalesced_until_flush(qtbot):
    """Incoming `podman logs` chunks are buffered and NOT written to the widget
    per chunk; one flush writes them all at once.

    This is the anti-freeze fix: during heavy generation the log follower fires
    readyRead very rapidly, and a per-chunk widget append floods the UI thread.
    _enqueue_log must defer to the flush timer so the widget updates at a bounded
    rate instead of once per chunk.
    """
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._enqueue_log("first line\n")
    w._enqueue_log("second line\n")
    # Deferred: nothing on the widget yet -- coalesced, not written per chunk.
    assert w.monitor_panel.log_view.toPlainText() == ""
    w._flush_log()
    text = w.monitor_panel.log_view.toPlainText()
    assert "first line" in text and "second line" in text


def test_flush_with_no_pending_is_a_noop(qtbot):
    """Flushing an empty buffer writes nothing (an idle server must not append
    blank lines every tick)."""
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._flush_log()
    assert w.monitor_panel.log_view.toPlainText() == ""


def test_router_model_switch_refetches(qtbot, monkeypatch):
    # Strict form (restored after the _status_timer teardown fix, 2026-08-07):
    # a one-shot iterator over the model-id sequence raises StopIteration on any
    # unplanned extra update_status() call, and a shared counter catches stray
    # fetch_props() calls. Both are safe now that a torn-down window's timer is
    # stopped and can't fire update_status() into this test.
    _ready(monkeypatch)
    monkeypatch.setattr(mw.MainWindow, "current_profile",
                        lambda self: _router_profile())
    monkeypatch.setattr(mw.MainWindow, "_router_host", lambda self, p: "127.0.0.1")
    ids = iter(["model-a", "model-a", "model-b"])
    monkeypatch.setattr(mw.MainWindow, "_router_pollable_model",
                        lambda self: next(ids))
    calls = {"n": 0}

    def _fetch(*a, **k):
        calls["n"] += 1
        return PropsInfo("b", 1, "a", 1, {})

    monkeypatch.setattr(mw.metrics, "fetch_props", _fetch)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w.update_status()   # model-a -> fetch
    assert w._props_model == "model-a"
    assert calls["n"] == 1
    w.update_status()   # model-a -> cached, no new fetch
    assert calls["n"] == 1
    w.update_status()   # model-b -> re-fetch
    assert w._props_model == "model-b"
    assert calls["n"] == 2


def _router_profile():
    from llama_launcher.core.spec import Profile, Runtime
    return Profile(name="r", image="img", mode="router", runtime=Runtime())
