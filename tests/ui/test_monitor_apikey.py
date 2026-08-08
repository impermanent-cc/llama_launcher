import llama_launcher.ui.main_window as mw
from llama_launcher.core.spec import Profile, Mount, Runtime


def _server(**settings):
    return Profile(name="Solo", image="img", runtime=Runtime(binary="podman"),
                   mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
                   model="/models/m.gguf", mode="server",
                   settings={"port": 8080, **settings})


def test_poll_api_key_server_uses_profile_api_key(qtbot):
    w = mw.MainWindow(); qtbot.addWidget(w)
    assert w._poll_api_key(_server(**{"api-key": "sk-secret"})) == "sk-secret"


def test_poll_api_key_server_none_when_unset(qtbot):
    w = mw.MainWindow(); qtbot.addWidget(w)
    assert w._poll_api_key(_server()) is None            # empty --api-key -> no header


def test_poll_api_key_router_uses_store(qtbot, monkeypatch):
    w = mw.MainWindow(); qtbot.addWidget(w)
    monkeypatch.setattr(mw.api_key_store, "read_api_key", lambda base, name: "sk-router")
    p = Profile(name="Host", image="img", mode="router")
    assert w._poll_api_key(p) == "sk-router"


def test_refresh_props_sends_server_api_key(qtbot, monkeypatch):
    # Regression: a single server started with --api-key was polled with NO key,
    # so /props came back 401 "Invalid API Key". The poll must send the key.
    seen = {}
    # capture the key, then return None so _refresh_props exits before set_props
    monkeypatch.setattr(mw.metrics, "fetch_props",
                        lambda port, timeout=1.0, api_key=None, host="127.0.0.1", **kw:
                        (seen.update(api_key=api_key) or None))
    w = mw.MainWindow(); qtbot.addWidget(w)
    w._props = None; w._props_model = None
    w._refresh_props(_server(**{"api-key": "sk-secret"}))
    assert seen.get("api_key") == "sk-secret"


def test_collect_monitor_data_sends_server_api_key(qtbot, monkeypatch):
    seen = {}
    monkeypatch.setattr(mw.metrics, "fetch_metrics",
                        lambda port, timeout=1.0, model=None, api_key=None, host="127.0.0.1", **kw:
                        (seen.update(api_key=api_key) or {}))
    monkeypatch.setattr(mw.metrics, "fetch_slots",
                        lambda port, timeout=1.0, model=None, api_key=None, host="127.0.0.1", **kw: [])
    w = mw.MainWindow(); qtbot.addWidget(w)
    w.load_profile(_server(**{"api-key": "sk-secret", "metrics": True}))
    w.collect_monitor_data()
    assert seen.get("api_key") == "sk-secret"
