import llama_launcher.ui.main_window as mw
import llama_launcher.ui.controllers.monitor_controller as mc
from llama_launcher.core.spec import Profile, Mount, Runtime
from llama_launcher.core.props import PropsInfo
from llama_launcher.ui.controllers.launch_controller import LaunchController
from llama_launcher.ui.controllers.monitor_controller import MonitorController
from llama_launcher.ui.panels.configure_panel import ConfigurePanel


def _profile():
    return Profile(name="m", image="img", runtime=Runtime(binary="podman"),
                   mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
                   model="/models/m.gguf", settings={"port": 8080, "metrics": True})


def test_on_stop_clears_log_follower_and_spawns_async_stop(qtbot, monkeypatch):
    """on_stop() kills the log follower immediately and spawns `podman stop`
    asynchronously (never blocking the UI thread) with the right argv."""
    spawned = {}
    # NOTE: patched on LaunchController (not MainWindow). Launch code now
    # calls self._spawn_async directly on the LaunchController instance (the
    # facade unwind repointed launch_controller.py's own calls off
    # self.window._spawn_async), so LaunchController is the class that must
    # be patched for this test to observe the call. tests/ui/conftest.py's
    # autouse _hermetic_ui_boundaries fixture ALSO class-patches
    # LaunchController._spawn_async (as a no-op); this same-symbol
    # class-patch still wins the override race because both use monkeypatch
    # and this test body's setattr runs after fixture setup (last write
    # wins, both undone at teardown).
    monkeypatch.setattr(LaunchController, "_spawn_async",
                        lambda self, argv, on_done=None: spawned.setdefault("argv", argv))
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(_profile())   # profile name "m" -> container llama-m

    killed = []

    class _FakeProc:
        def kill(self):
            killed.append(True)

    w._monitor._log_proc = _FakeProc()
    w._launch.on_stop()
    assert w._monitor._log_proc is None
    assert killed == [True]
    assert spawned["argv"] == ["podman", "stop", "-t", "10", "llama-m"]


def test_on_stop_uses_profile_stop_timeout(qtbot, monkeypatch):
    """The Stop button's `podman stop -t` grace period comes from the profile's
    configurable stop_timeout, not a hardcoded 10s."""
    spawned = {}
    monkeypatch.setattr(LaunchController, "_spawn_async",
                        lambda self, argv, on_done=None: spawned.setdefault("argv", argv))
    w = mw.MainWindow()
    qtbot.addWidget(w)
    p = _profile()
    p.runtime.stop_timeout = 25
    w._configure_panel.load_profile(p)
    w._launch.on_stop()
    assert spawned["argv"] == ["podman", "stop", "-t", "25", "llama-m"]


def test_on_restart_uses_profile_stop_timeout(qtbot, monkeypatch):
    """Restart tears down with the same configurable grace period as Stop."""
    spawned = {}
    monkeypatch.setattr(LaunchController, "_spawn_async",
                        lambda self, argv, on_done=None: spawned.setdefault("argv", argv))
    w = mw.MainWindow()
    qtbot.addWidget(w)
    p = _profile()
    p.runtime.stop_timeout = 30
    w._configure_panel.load_profile(p)
    w._launch.on_restart()
    assert spawned["argv"] == ["podman", "stop", "-t", "30", "llama-m"]


def test_collect_monitor_data(qtbot, monkeypatch):
    monkeypatch.setattr(mw.metrics, "fetch_metrics",
                        lambda port, timeout=1.0, **kw: {"llamacpp:predicted_tokens_seconds": 50.0,
                                                         "llamacpp:prompt_tokens_seconds": 200.0})
    monkeypatch.setattr(mw.metrics, "fetch_slots", lambda port, timeout=1.0, **kw:
                        [{"n_ctx": 100, "n_prompt_tokens_processed": 40}])
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda ssh_target="": [])
    monkeypatch.setattr(mw.runtime, "stats",
                        lambda name, b, connection="": {"cpu_perc": "9%", "mem_usage": "1G / 16G"})
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(_profile())
    d = w._monitor.collect_monitor_data()
    assert d["tok_s"] == 50.0 and d["prompt_tok_s"] == 200.0
    assert abs(d["kv_pct"] - 0.40) < 1e-9
    assert d["metrics_on"] is True
    assert d["cpu"] == "9%"


def test_collect_monitor_data_reports_speculating(qtbot, monkeypatch):
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    monkeypatch.setattr(mw.metrics, "fetch_slots",
                        lambda *a, **k: [{"speculative": True, "n_ctx": 4096}])
    monkeypatch.setattr(mw.metrics, "fetch_metrics", lambda *a, **k: {})
    monkeypatch.setattr(mw.runtime, "stats", lambda name, binary, connection="": {})
    monkeypatch.setattr(mw.runtime, "started_at", lambda name, binary, connection="": None)
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda ssh_target="": [])
    w = mw.MainWindow()
    qtbot.addWidget(w)
    assert w._monitor.collect_monitor_data()["speculating"] is True


def test_compute_monitor_target_carries_decode_prev(qtbot, monkeypatch):
    """The target snapshot must carry the previous decode read so the off-thread
    gather can compute the live rate against it."""
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._monitor._decode_prev = (42.0, 3.0)
    t = w._monitor._compute_monitor_target(running=True)
    assert t["decode_prev"] == (42.0, 3.0)


def test_update_status_advances_decode_prev_from_result(qtbot, monkeypatch):
    """Each rendered gather result advances _decode_prev to its decode_now, so
    the next tick measures the rate over the interval between the two reads."""
    _ready(monkeypatch)
    monkeypatch.setattr(mw.metrics, "fetch_props", lambda *a, **k: None)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._monitor._monitor_inflight = True     # block a fresh async gather from overwriting
    w._monitor._monitor_result = {"decode_now": (9.0, 2.0), "metrics_on": True}
    w._monitor.update_status()
    assert w._monitor._decode_prev == (9.0, 2.0)


def test_decode_prev_cleared_when_not_running(qtbot, monkeypatch):
    """A stopped server clears the decode baseline so the first rate after the
    next start isn't computed across the downtime (a huge bogus spike)."""
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    monkeypatch.setattr(mw.runtime, "container_state", lambda *a, **k: "stopped")
    monkeypatch.setattr(mw.health, "probe_health", lambda *a, **k: "down")
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._monitor._decode_prev = (5.0, 1.0)
    w._monitor.update_status()
    assert w._monitor._decode_prev is None


def _ready(monkeypatch):
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    monkeypatch.setattr(mw.runtime, "container_state", lambda name, binary, connection="": "running")
    monkeypatch.setattr(mw.health, "probe_health", lambda port, **kw: "ready")
    monkeypatch.setattr(MonitorController, "_log_follower_active", lambda self: True)
    monkeypatch.setattr(MonitorController, "_update_spec_stats", lambda self, p: None)


def test_props_fetched_once_when_ready(qtbot, monkeypatch):
    _ready(monkeypatch)
    calls = []
    monkeypatch.setattr(mw.metrics, "fetch_props",
                        lambda *a, **k: calls.append(1) or PropsInfo("b", 1, "a", 1, {}))
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._monitor.update_status()
    w._monitor.update_status()
    assert len(calls) == 1                 # cached after first success


def test_props_not_fetched_while_loading(qtbot, monkeypatch):
    _ready(monkeypatch)
    monkeypatch.setattr(mw.health, "probe_health", lambda port, **kw: "loading")
    calls = []
    monkeypatch.setattr(mw.metrics, "fetch_props", lambda *a, **k: calls.append(1))
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._monitor.update_status()
    assert calls == []                     # only fetch once the model is ready


def test_failed_props_fetch_retries_next_poll(qtbot, monkeypatch):
    _ready(monkeypatch)
    calls = []
    monkeypatch.setattr(mw.metrics, "fetch_props",
                        lambda *a, **k: calls.append(1) or None)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._monitor.update_status()
    w._monitor.update_status()
    assert len(calls) == 2                 # None result is not cached; retries


def test_build_monitor_data_gathers_from_a_plain_target(monkeypatch):
    """build_monitor_data() is a pure function of a primitives-only target so it
    can run off the UI thread: it must gather from those primitives, never from
    live widget/profile state."""
    monkeypatch.setattr(mw.metrics, "fetch_metrics",
                        lambda *a, **k: {"llamacpp:predicted_tokens_seconds": 50.0})
    monkeypatch.setattr(mw.metrics, "fetch_slots",
                        lambda *a, **k: [{"n_ctx": 100, "n_prompt_tokens_processed": 40}])
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda ssh_target="": [])
    monkeypatch.setattr(mw.runtime, "stats",
                        lambda name, b, connection="": {"cpu_perc": "9%", "mem_usage": "1G / 16G"})
    monkeypatch.setattr(mw.runtime, "started_at", lambda name, b, connection="": None)
    target = {"running": True, "port": 8080, "metrics_on": True, "host": "127.0.0.1",
              "key": None, "model_scope": None, "poll": True,
              "name": "llama-x", "binary": "podman"}
    d = mw.build_monitor_data(target)
    assert d["tok_s"] == 50.0 and d["cpu"] == "9%"
    assert abs(d["kv_pct"] - 0.40) < 1e-9


def test_build_monitor_data_computes_live_gen_tok_s_from_decode_delta(monkeypatch):
    """gen tok/s must be the live n_decode_total delta rate: llama.cpp's
    predicted_tokens_seconds gauge reads 0 during an in-flight generation, so
    the counter delta between two polls is the only live throughput signal."""
    monkeypatch.setattr(mw.metrics, "fetch_metrics",
                        lambda *a, **k: {"llamacpp:n_decode_total": 123.0,
                                         "llamacpp:predicted_tokens_seconds": 0.0})
    monkeypatch.setattr(mw.metrics, "fetch_slots", lambda *a, **k: [])
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda ssh_target="": [])
    monkeypatch.setattr(mw.runtime, "stats", lambda *a, **k: {})
    monkeypatch.setattr(mw.runtime, "started_at", lambda *a, **k: None)
    monkeypatch.setattr(mc.time, "monotonic", lambda: 11.0)
    target = {"running": True, "port": 8080, "metrics_on": True, "host": "127.0.0.1",
              "key": None, "model_scope": None, "poll": True,
              "name": "llama-x", "binary": "podman",
              "decode_prev": (100.0, 10.0)}          # 100 decodes at t=10.0
    d = mw.build_monitor_data(target)
    assert abs(d["gen_tok_s_live"] - 23.0) < 1e-9    # (123-100)/(11.0-10.0)
    assert d["decode_now"] == (123.0, 11.0)          # handed to the next tick as prev


def test_build_monitor_data_live_gen_none_on_first_poll(monkeypatch):
    """With no prior decode read, there's no rate yet -- but the current read is
    still returned so the NEXT poll can compute one."""
    monkeypatch.setattr(mw.metrics, "fetch_metrics",
                        lambda *a, **k: {"llamacpp:n_decode_total": 77.0})
    monkeypatch.setattr(mw.metrics, "fetch_slots", lambda *a, **k: [])
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda ssh_target="": [])
    monkeypatch.setattr(mw.runtime, "stats", lambda *a, **k: {})
    monkeypatch.setattr(mw.runtime, "started_at", lambda *a, **k: None)
    monkeypatch.setattr(mc.time, "monotonic", lambda: 5.0)
    target = {"running": True, "port": 8080, "metrics_on": True, "host": "127.0.0.1",
              "key": None, "model_scope": None, "poll": True,
              "name": "llama-x", "binary": "podman"}   # no decode_prev
    d = mw.build_monitor_data(target)
    assert d["gen_tok_s_live"] is None
    assert d["decode_now"] == (77.0, 5.0)


def test_build_monitor_data_prefers_kv_cache_metric(monkeypatch):
    """KV% uses llama.cpp's kv_cache_usage_ratio gauge when present, not the
    slots-derived prompt-token estimate (which reads 0 when idle)."""
    monkeypatch.setattr(mw.metrics, "fetch_metrics",
                        lambda *a, **k: {"llamacpp:kv_cache_usage_ratio": 0.83})
    monkeypatch.setattr(mw.metrics, "fetch_slots",
                        lambda *a, **k: [{"n_ctx": 100, "n_prompt_tokens_processed": 40}])
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda ssh_target="": [])
    monkeypatch.setattr(mw.runtime, "stats", lambda name, b, connection="": {})
    monkeypatch.setattr(mw.runtime, "started_at", lambda name, b, connection="": None)
    target = {"running": True, "port": 8080, "metrics_on": True, "host": "127.0.0.1",
              "key": None, "model_scope": None, "poll": True,
              "name": "llama-x", "binary": "podman"}
    d = mw.build_monitor_data(target)
    assert d["kv_pct"] == 0.83     # the gauge, not 0.40 from slots


def test_build_monitor_data_returns_none_when_not_running(monkeypatch):
    """A not-running target does no I/O and returns None, so the worker emits
    nothing (an idle/stopped server isn't polled off-thread every second)."""
    called = []
    monkeypatch.setattr(mw.runtime, "stats", lambda *a, **k: called.append(1))
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda ssh_target="": called.append(1))
    assert mw.build_monitor_data({"running": False}) is None
    assert called == []


def test_build_monitor_data_threads_mon_conn_into_stats_and_started_at(monkeypatch):
    """A target carrying mon_conn (the monitored instance's node connection)
    must thread it into runtime.stats/started_at -- otherwise a remote
    instance's CPU/mem/uptime are read from LOCAL podman instead of the node
    the container actually runs on."""
    monkeypatch.setattr(mw.metrics, "fetch_metrics", lambda *a, **k: {})
    monkeypatch.setattr(mw.metrics, "fetch_slots", lambda *a, **k: [])
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda ssh_target="": [])
    stats_calls = []
    started_calls = []
    monkeypatch.setattr(mw.runtime, "stats",
                        lambda name, b, connection="": stats_calls.append(connection) or {})
    monkeypatch.setattr(mw.runtime, "started_at",
                        lambda name, b, connection="": started_calls.append(connection) or None)
    target = {"running": True, "port": 8080, "metrics_on": False, "host": "10.0.0.2",
              "key": None, "model_scope": None, "poll": True,
              "name": "llama-rem", "binary": "podman", "mon_conn": "box-b"}
    mw.build_monitor_data(target)
    assert stats_calls == ["box-b"]
    assert started_calls == ["box-b"]


def test_compute_monitor_target_uses_node_host_and_connection_for_remote(qtbot, monkeypatch):
    """A focused REMOTE instance's monitor target must poll the node's real
    host (not 127.0.0.1 from dial_host(0.0.0.0)) and carry the node's
    connection so container_state/stats/started_at hit the right podman."""
    from llama_launcher.core.instances import Instance
    from llama_launcher.core.nodes import Node
    from llama_launcher.store.nodes import add_node
    w = mw.MainWindow()
    qtbot.addWidget(w)
    add_node(Node(name="box-b", kind="remote", connection="box-b",
                  ssh_target="me@10.0.0.2"), w.base_dir())
    w._configure_panel.load_profile(Profile(name="Solo", image="img", settings={"port": 8080}))
    w._monitor._instances = [Instance(
        name="llama-rem", profile="rem", mode="server", running=True,
        port=8081, host="0.0.0.0", embeddings=False, reranking=False,
        stop_timeout=10, binary="podman", node="box-b")]
    w._monitor._active_instance = w._monitor._instances[0]
    target = w._monitor._compute_monitor_target(running=True)
    assert target["host"] == "10.0.0.2"        # node's dial host, not 127.0.0.1
    assert target["mon_conn"] == "box-b"


def test_compute_monitor_target_local_stays_unchanged(qtbot):
    """A local instance's target keeps its old host/connection shape exactly:
    dial_host()-derived loopback and an empty mon_conn."""
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(_profile())
    target = w._monitor._compute_monitor_target(running=True)
    assert target["host"] == "127.0.0.1"
    assert target["mon_conn"] == ""


def test_update_status_populates_monitor_target_off_ui_thread(qtbot, monkeypatch):
    """update_status snapshots the poll inputs into _monitor_target on the UI
    thread; the worker reads that plain dict and does the blocking gather. When
    the container is running the target is marked running=True."""
    _ready(monkeypatch)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._monitor.update_status()
    assert w._monitor._monitor_target.get("running") is True


def test_update_status_marks_target_not_running_when_stopped(qtbot, monkeypatch):
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    monkeypatch.setattr(mw.runtime, "container_state", lambda name, binary, connection="": "stopped")
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._monitor.update_status()
    assert w._monitor._monitor_target.get("running") is False


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
    w._monitor.update_status()                       # dispatch; _monitor_result still None
    from PySide6.QtCore import QThreadPool
    QThreadPool.globalInstance().waitForDone(2000)
    w._monitor.update_status()                       # renders the gathered result
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
    w._monitor._enqueue_log("first line\n")
    w._monitor._enqueue_log("second line\n")
    # Deferred: nothing on the widget yet -- coalesced, not written per chunk.
    assert w.monitor_panel.log_view.toPlainText() == ""
    w._monitor._flush_log()
    text = w.monitor_panel.log_view.toPlainText()
    assert "first line" in text and "second line" in text


def test_flush_with_no_pending_is_a_noop(qtbot):
    """Flushing an empty buffer writes nothing (an idle server must not append
    blank lines every tick)."""
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._monitor._flush_log()
    assert w.monitor_panel.log_view.toPlainText() == ""


def test_router_model_switch_refetches(qtbot, monkeypatch):
    # Strict form (restored after the _status_timer teardown fix, 2026-08-07):
    # a one-shot iterator over the model-id sequence raises StopIteration on any
    # unplanned extra update_status() call, and a shared counter catches stray
    # fetch_props() calls. Both are safe now that a torn-down window's timer is
    # stopped and can't fire update_status() into this test.
    _ready(monkeypatch)
    monkeypatch.setattr(ConfigurePanel, "current_profile",
                        lambda self: _router_profile())
    monkeypatch.setattr(MonitorController, "_router_host", lambda self, p: "127.0.0.1")
    ids = iter(["model-a", "model-a", "model-b"])
    monkeypatch.setattr(MonitorController, "_router_pollable_model",
                        lambda self: next(ids))
    calls = {"n": 0}

    def _fetch(*a, **k):
        calls["n"] += 1
        return PropsInfo("b", 1, "a", 1, {})

    monkeypatch.setattr(mw.metrics, "fetch_props", _fetch)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._monitor.update_status()   # model-a -> fetch
    assert w._monitor._props_model == "model-a"
    assert calls["n"] == 1
    w._monitor.update_status()   # model-a -> cached, no new fetch
    assert calls["n"] == 1
    w._monitor.update_status()   # model-b -> re-fetch
    assert w._monitor._props_model == "model-b"
    assert calls["n"] == 2


def test_on_launch_native_spawns_process_not_container(qtbot, monkeypatch):
    from llama_launcher.services import native as native_svc
    from llama_launcher.services.native import NativeResult
    calls = {}
    monkeypatch.setattr(native_svc, "launch_native",
                        lambda p, base, now_iso: calls.setdefault(
                            "res", NativeResult(True, "llama-nat", "127.0.0.1", 8080, 4242)))
    # No live instance for this profile -- the double-launch guard (Fix 1)
    # must let the launch proceed.
    monkeypatch.setattr(native_svc, "list_native_instances", lambda base_dir: [])
    # Fail the test loudly if the container path is taken instead.
    monkeypatch.setattr(LaunchController, "_spawn_async",
                        lambda self, *a, **k: calls.setdefault("container", True))
    w = mw.MainWindow()
    qtbot.addWidget(w)
    p = _profile()
    p.runtime.launch_mode = "native"
    p.runtime.native_binary = "/opt/bin/llama-server"
    w._configure_panel.load_profile(p)
    # NOTE (deviation from task-8-brief.md): ConfigurePanel does not yet
    # round-trip Runtime.launch_mode/native_binary through load_profile() /
    # current_profile() -- that widget wiring is Task 11's scope and has not
    # landed in this worktree (only Tasks 1-7 are merged ahead of this one).
    # current_profile() is the actual interface on_launch() consumes, so
    # patch it directly to return our native profile rather than relying on
    # a UI round-trip that doesn't exist yet.
    monkeypatch.setattr(w._configure_panel, "current_profile", lambda: p)
    # bypass VRAM/validation dialogs
    monkeypatch.setattr(w._launch, "_validate_or_warn", lambda: True)
    monkeypatch.setattr(w._launch, "vram_check", lambda: "")
    w._launch.on_launch()
    assert "res" in calls and "container" not in calls


def test_on_launch_native_refuses_when_already_running(qtbot, monkeypatch):
    """Fix 1: relaunching a native profile that already has a live instance
    must NOT spawn a second llama-server -- doing so would fail to bind the
    in-use port and orphan the original process (registry entry overwritten
    with the dead PID, original left running with no way to stop it)."""
    from llama_launcher.services import native as native_svc

    p = _profile()
    p.runtime.launch_mode = "native"
    p.runtime.native_binary = "/opt/bin/llama-server"

    launch_calls = {"n": 0}
    monkeypatch.setattr(native_svc, "launch_native",
                        lambda p, base, now_iso: launch_calls.__setitem__(
                            "n", launch_calls["n"] + 1))
    monkeypatch.setattr(
        native_svc, "list_native_instances",
        lambda base_dir: [{"name": native_svc.native_name(p.name),
                           "running": True, "profile": p.name,
                           "mode": "server", "pid": 4242, "kind": "native"}])

    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(p)
    monkeypatch.setattr(w._configure_panel, "current_profile", lambda: p)
    monkeypatch.setattr(w._launch, "_validate_or_warn", lambda: True)
    monkeypatch.setattr(w._launch, "vram_check", lambda: "")

    errors = []
    monkeypatch.setattr(w._launch, "_report_launch_error",
                        lambda *a, **k: errors.append((a, k)))

    w._launch.on_launch()

    assert launch_calls["n"] == 0
    assert errors, "_report_launch_error should have fired to refuse the launch"


def test_on_launch_foreground_reports_when_no_terminal(qtbot, monkeypatch):
    """A foreground container launch on a box with no installed terminal (e.g.
    konsole absent on GNOME) must surface a clear error, not crash on_launch
    with an unhandled FileNotFoundError."""
    from llama_launcher.services import terminal as term_mod

    def _raise(*a, **k):
        raise term_mod.NoTerminalError("no terminal emulator found")
    monkeypatch.setattr(term_mod, "launch", _raise)

    p = _profile()  # container, server, detached=False -> foreground terminal path
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(p)
    monkeypatch.setattr(w._configure_panel, "current_profile", lambda: p)
    monkeypatch.setattr(w._launch, "_validate_or_warn", lambda: True)
    monkeypatch.setattr(w._launch, "vram_check", lambda: "")

    errors = []
    monkeypatch.setattr(w._launch, "_report_launch_error",
                        lambda *a, **k: errors.append((a, k)))

    w._launch.on_launch()  # must not raise

    assert errors, "_report_launch_error should fire when no terminal is available"


def _router_profile():
    from llama_launcher.core.spec import Profile, Runtime
    return Profile(name="r", image="img", mode="router", runtime=Runtime())


def test_build_monitor_data_computes_live_prompt_tok_s_from_slot_delta(monkeypatch):
    """prompt tok/s must be live during prefill: the prompt_tokens_seconds
    gauge only updates at request completion (0 through a 27k-token prefill),
    but the processing slot's n_prompt_tokens_processed grows batch by batch,
    so its delta between two polls is a live prefill rate."""
    monkeypatch.setattr(mw.metrics, "fetch_metrics", lambda *a, **k: {})
    monkeypatch.setattr(mw.metrics, "fetch_slots",
                        lambda *a, **k: [{"is_processing": True, "n_ctx": 32768,
                                          "n_prompt_tokens_processed": 5000}])
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda ssh_target="": [])
    monkeypatch.setattr(mw.runtime, "stats", lambda *a, **k: {})
    monkeypatch.setattr(mw.runtime, "started_at", lambda *a, **k: None)
    monkeypatch.setattr(mc.time, "monotonic", lambda: 11.0)
    target = {"running": True, "port": 8080, "metrics_on": True, "host": "127.0.0.1",
              "key": None, "model_scope": None, "poll": True,
              "name": "llama-x", "binary": "podman",
              "prompt_prev": (1000, 10.0)}
    d = mw.build_monitor_data(target)
    assert abs(d["prompt_tok_s_live"] - 4000.0) < 1e-9   # (5000-1000)/1.0
    assert d["prompt_now"] == (5000, 11.0)


def test_build_monitor_data_prompt_live_none_when_not_prefilling(monkeypatch):
    monkeypatch.setattr(mw.metrics, "fetch_metrics", lambda *a, **k: {})
    monkeypatch.setattr(mw.metrics, "fetch_slots",
                        lambda *a, **k: [{"is_processing": False,
                                          "n_prompt_tokens_processed": 5000}])
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda ssh_target="": [])
    monkeypatch.setattr(mw.runtime, "stats", lambda *a, **k: {})
    monkeypatch.setattr(mw.runtime, "started_at", lambda *a, **k: None)
    target = {"running": True, "port": 8080, "metrics_on": True, "host": "127.0.0.1",
              "key": None, "model_scope": None, "poll": True,
              "name": "llama-x", "binary": "podman",
              "prompt_prev": (1000, 10.0)}
    d = mw.build_monitor_data(target)
    assert d["prompt_tok_s_live"] is None
    assert d["prompt_now"] is None


def test_compute_monitor_target_carries_prompt_prev(qtbot):
    w = mw.MainWindow(); qtbot.addWidget(w)
    w._monitor._prompt_prev = (1000, 10.0)
    t = w._monitor._compute_monitor_target(running=True)
    assert t["prompt_prev"] == (1000, 10.0)
