"""LaunchController: rpc launch_mode routes on_launch/on_stop through the
services.rpc pool orchestrator instead of the single-container path."""
from llama_launcher.core.spec import Profile, Runtime


def _rpc_profile():
    return Profile(name="pool", image="img:tag", runtime=Runtime(launch_mode="rpc"))


def test_on_launch_rpc_calls_launch_pool(main_window, monkeypatch):
    ctl = main_window._launch
    main_window._configure_panel.load_profile(_rpc_profile())
    monkeypatch.setattr(ctl, "_validate_or_warn", lambda: True)

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

    import llama_launcher.services.rpc as rpc
    monkeypatch.setattr(
        rpc, "launch_pool", lambda p, base, **k: rpc.PoolResult(False, "worker 0 failed"))
    reported = {}
    monkeypatch.setattr(
        ctl, "_report_launch_error",
        lambda text=None, *, show_dialog=False: reported.setdefault("text", (text, show_dialog)))

    ctl.on_launch()

    assert reported["text"] == ("worker 0 failed", True)


def test_on_stop_rpc_calls_stop_pool(main_window, monkeypatch):
    ctl = main_window._launch
    main_window._configure_panel.load_profile(_rpc_profile())

    import llama_launcher.services.rpc as rpc
    called = {}
    monkeypatch.setattr(
        rpc, "stop_pool", lambda p, base, **k: called.setdefault("ok", True))

    ctl.on_stop()

    assert called.get("ok")
