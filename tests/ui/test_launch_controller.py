"""LaunchController: rpc launch_mode routes on_launch/on_stop through the
services.rpc pool orchestrator instead of the single-container path.

The pool orchestrator (rpc.launch_pool / rpc.stop_pool) runs OFF the UI thread
via `_run_pool_async`, whose completion is marshalled back to the UI thread.
Tests install a synchronous seam -- `_run_pool_async(work, on_done)` ->
`on_done(work())` -- so the routing is exercised without real threading, plus
one integration test that the real seam runs work off-thread and delivers the
result on the UI thread.
"""
import threading

from PySide6.QtCore import QCoreApplication, QThreadPool

import llama_launcher.services.runtime as _runtime
from llama_launcher.core.spec import Profile, Runtime


def _rpc_profile():
    return Profile(name="pool", image="img:tag", runtime=Runtime(launch_mode="rpc"))


def _sync_seam(ctl, monkeypatch):
    """Run the pool orchestrator inline and deliver its result immediately,
    standing in for the off-thread dispatch."""
    monkeypatch.setattr(ctl, "_run_pool_async",
                        lambda work, on_done: on_done(work()))


def test_on_launch_rpc_calls_launch_pool(main_window, monkeypatch):
    ctl = main_window._launch
    main_window._configure_panel.load_profile(_rpc_profile())
    monkeypatch.setattr(ctl, "_validate_or_warn", lambda: True)
    _sync_seam(ctl, monkeypatch)

    import llama_launcher.services.rpc as rpc
    called = {}

    def _launch_pool(p, base, **k):
        called["ok"] = True
        return rpc.PoolResult(True)

    monkeypatch.setattr(rpc, "launch_pool", _launch_pool)

    ctl.on_launch()

    assert called.get("ok")


def test_on_launch_rpc_reports_error_on_failure(main_window, monkeypatch):
    ctl = main_window._launch
    main_window._configure_panel.load_profile(_rpc_profile())
    monkeypatch.setattr(ctl, "_validate_or_warn", lambda: True)
    _sync_seam(ctl, monkeypatch)

    import llama_launcher.services.rpc as rpc
    monkeypatch.setattr(
        rpc, "launch_pool", lambda p, base, **k: rpc.PoolResult(False, "worker 0 failed"))
    reported = {}
    monkeypatch.setattr(
        ctl, "_report_launch_error",
        lambda text=None, *, show_dialog=False: reported.setdefault("text", (text, show_dialog)))

    ctl.on_launch()

    assert reported["text"] == ("worker 0 failed", True)


def test_on_launch_rpc_refuses_when_pool_already_running(main_window, monkeypatch):
    """A second Launch click over a LIVE pool must not call rpc.launch_pool --
    that would tear down the current pool's live ssh tunnels and then fail on
    the worker container name collision, degrading a healthy pool. It must
    refuse instead, the way the native branch already does."""
    ctl = main_window._launch
    main_window._configure_panel.load_profile(_rpc_profile())
    monkeypatch.setattr(ctl, "_validate_or_warn", lambda: True)
    monkeypatch.setattr(_runtime, "container_state",
                        lambda name, binary, connection="": "running")

    import llama_launcher.services.rpc as rpc
    called = {}
    monkeypatch.setattr(
        rpc, "launch_pool", lambda p, base, **k: called.setdefault("ok", True))
    reported = {}
    monkeypatch.setattr(
        ctl, "_report_launch_error",
        lambda text=None, *, show_dialog=False: reported.setdefault("text", (text, show_dialog)))

    ctl.on_launch()

    assert "ok" not in called, "rpc.launch_pool must not be called over a live pool"
    text, show_dialog = reported["text"]
    assert "already running" in text
    assert show_dialog is True


def test_on_stop_rpc_calls_stop_pool(main_window, monkeypatch):
    ctl = main_window._launch
    main_window._configure_panel.load_profile(_rpc_profile())
    _sync_seam(ctl, monkeypatch)

    import llama_launcher.services.rpc as rpc
    called = {}
    monkeypatch.setattr(
        rpc, "stop_pool", lambda p, base, **k: called.setdefault("ok", True))

    ctl.on_stop()

    assert called.get("ok")


# -- off-thread dispatch --------------------------------------------------------

def test_launch_pool_defers_orchestrator_to_run_pool_async(main_window, monkeypatch):
    """on_launch must not run rpc.launch_pool inline on the UI thread -- it must
    hand a callable to _run_pool_async (which runs it off-thread)."""
    ctl = main_window._launch
    main_window._configure_panel.load_profile(_rpc_profile())
    monkeypatch.setattr(ctl, "_validate_or_warn", lambda: True)

    import llama_launcher.services.rpc as rpc
    ran = {"launch": False}
    monkeypatch.setattr(
        rpc, "launch_pool",
        lambda p, base, **k: (ran.__setitem__("launch", True), rpc.PoolResult(True))[1])

    captured = {}
    monkeypatch.setattr(ctl, "_run_pool_async",
                        lambda work, on_done: captured.update(work=work, on_done=on_done))

    ctl.on_launch()

    assert "work" in captured, "on_launch must dispatch through _run_pool_async"
    assert ran["launch"] is False, "orchestrator must not run inline on the UI thread"
    # the deferred callable is what the worker thread runs
    res = captured["work"]()
    assert ran["launch"] is True
    assert res.ok


def test_launch_pool_sets_inflight_and_starting_status(main_window, monkeypatch):
    ctl = main_window._launch
    main_window._configure_panel.load_profile(_rpc_profile())
    monkeypatch.setattr(ctl, "_validate_or_warn", lambda: True)
    monkeypatch.setattr(ctl, "_run_pool_async", lambda work, on_done: None)

    ctl.on_launch()

    assert ctl._pool_inflight is True
    assert "starting" in main_window.status_label.text().lower()


def test_launch_pool_reentrant_guard_blocks_second_dispatch(main_window, monkeypatch):
    """While a pool op is in-flight, a second Launch must not fire another
    orchestrator run (the container-name guard doesn't cover the window before
    the head container is up)."""
    ctl = main_window._launch
    main_window._configure_panel.load_profile(_rpc_profile())
    monkeypatch.setattr(ctl, "_validate_or_warn", lambda: True)
    monkeypatch.setattr(_runtime, "container_state",
                        lambda name, binary, connection="": "stopped")
    _sync_seam(ctl, monkeypatch)
    ctl._pool_inflight = True

    import llama_launcher.services.rpc as rpc
    called = {}
    monkeypatch.setattr(rpc, "launch_pool", lambda p, base, **k: called.setdefault("x", True))

    ctl.on_launch()

    assert "x" not in called, "in-flight guard must block a second orchestrator run"


def test_on_pool_result_success_updates_status_and_clears_inflight(main_window, monkeypatch):
    ctl = main_window._launch
    ctl._pool_inflight = True
    updated = {}
    monkeypatch.setattr(main_window._monitor, "update_status",
                        lambda: updated.setdefault("x", True))

    import llama_launcher.services.rpc as rpc
    ctl._on_pool_result(rpc.PoolResult(True))

    assert updated.get("x")
    assert ctl._pool_inflight is False


def test_on_pool_result_failure_reports_error_and_clears_inflight(main_window, monkeypatch):
    ctl = main_window._launch
    ctl._pool_inflight = True
    reported = {}
    monkeypatch.setattr(
        ctl, "_report_launch_error",
        lambda text=None, *, show_dialog=False: reported.setdefault("t", (text, show_dialog)))
    monkeypatch.setattr(main_window._monitor, "update_status",
                        lambda: reported.setdefault("upd", True))

    import llama_launcher.services.rpc as rpc
    ctl._on_pool_result(rpc.PoolResult(False, "boom"))

    assert reported["t"] == ("boom", True)
    assert "upd" not in reported, "a failed launch must not report success via update_status"
    assert ctl._pool_inflight is False


def test_on_stop_rpc_defers_to_run_pool_async(main_window, monkeypatch):
    ctl = main_window._launch
    main_window._configure_panel.load_profile(_rpc_profile())

    import llama_launcher.services.rpc as rpc
    ran = {"stop": False}
    monkeypatch.setattr(rpc, "stop_pool",
                        lambda p, base, **k: ran.__setitem__("stop", True))
    captured = {}
    monkeypatch.setattr(ctl, "_run_pool_async",
                        lambda work, on_done: captured.update(work=work, on_done=on_done))

    ctl.on_stop()

    assert "work" in captured
    assert ran["stop"] is False, "stop_pool must not run inline on the UI thread"
    captured["work"]()
    assert ran["stop"] is True


def test_on_stop_rpc_sets_stopping_status_and_inflight(main_window, monkeypatch):
    ctl = main_window._launch
    main_window._configure_panel.load_profile(_rpc_profile())
    monkeypatch.setattr(ctl, "_run_pool_async", lambda work, on_done: None)

    ctl.on_stop()

    assert ctl._pool_inflight is True
    assert "stopping" in main_window.status_label.text().lower()


def test_on_pool_stopped_updates_status_and_clears_inflight(main_window, monkeypatch):
    ctl = main_window._launch
    ctl._pool_inflight = True
    updated = {}
    monkeypatch.setattr(main_window._monitor, "update_status",
                        lambda: updated.setdefault("x", True))

    ctl._on_pool_stopped(None)

    assert updated.get("x")
    assert ctl._pool_inflight is False


def test_run_pool_async_runs_work_off_thread_and_delivers_on_ui_thread(main_window):
    """The real seam: work() runs on a pool thread; on_done is delivered back on
    the UI (caller) thread via a queued signal."""
    ctl = main_window._launch
    ui_thread = threading.get_ident()
    info = {}

    def work():
        info["work_thread"] = threading.get_ident()
        return "RESULT"

    delivered = {}

    def on_done(res):
        delivered["res"] = res
        delivered["done_thread"] = threading.get_ident()

    ctl._run_pool_async(work, on_done)

    QThreadPool.globalInstance().waitForDone(3000)
    for _ in range(200):
        if "res" in delivered:
            break
        QCoreApplication.processEvents()

    assert delivered.get("res") == "RESULT"
    assert info["work_thread"] != ui_thread, "work must run off the UI thread"
    assert delivered["done_thread"] == ui_thread, "on_done must run on the UI thread"


def test_run_pool_async_signaller_not_parented_to_window(main_window):
    """The result carrier must NOT be a child of the window: a worker still
    running past the drain ceiling (e.g. blocked on an unreachable node ~55s)
    would otherwise emit onto a C++ object deleted with the window."""
    ctl = main_window._launch
    ctl._run_pool_async(lambda: "x", lambda res: None)
    assert ctl._pool_signaller is not None
    assert ctl._pool_signaller.parent() is None


def test_run_pool_async_delivers_worker_exception_as_failure(main_window):
    """A raise inside the orchestrator must not escape the pool thread -- it is
    delivered to on_done as a failed PoolResult."""
    ctl = main_window._launch

    def work():
        raise RuntimeError("kaboom")

    delivered = {}
    ctl._run_pool_async(work, lambda res: delivered.setdefault("res", res))

    QThreadPool.globalInstance().waitForDone(3000)
    for _ in range(200):
        if "res" in delivered:
            break
        QCoreApplication.processEvents()

    res = delivered["res"]
    assert res.ok is False
    assert "kaboom" in res.error


def test_drain_awaits_inflight_pool_worker(main_window):
    """LaunchController.drain() must wait for its own in-flight pool worker to
    finish rather than leaving it running against a torn-down window."""
    ctl = main_window._launch
    proceed = threading.Event()
    ran = {"done": False}

    def work():
        proceed.wait(3.0)
        ran["done"] = True
        return "ok"

    ctl._run_pool_async(work, lambda res: None)
    # Release the worker shortly after drain() starts blocking on it.
    threading.Timer(0.1, proceed.set).start()

    ctl.drain()

    assert ran["done"] is True, "drain() must wait for the in-flight pool worker"
