import llama_launcher.ui.main_window as mw
from llama_launcher.core.spec import Profile, Mount, Runtime


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
