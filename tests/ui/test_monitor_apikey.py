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


def test_monitored_router_instance_resolves_router_key(qtbot, tmp_path, monkeypatch):
    # Monitoring a running ROUTER via the instances row when the stored profile
    # is missing or was saved as a *server* (mode drift) must resolve the router
    # file key from the instance's identity, not fall back to the form (which
    # would send no key -> 401 Invalid API Key).
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from llama_launcher.core.instances import Instance
    from llama_launcher.services import api_key as ak
    w = mw.MainWindow(); qtbot.addWidget(w)
    ak.ensure_api_key(w.router_base_dir(), "Gemma")          # key file as at launch
    # stored profile is SERVER mode; the running container is a ROUTER
    from llama_launcher.store import profiles as store
    store.save_profile(Profile(name="Gemma", image="img", mode="server",
                               settings={"port": 8080, "api-key": "sk-wrong"}),
                       w.router_base_dir())
    w._active_instance = Instance(name="llama-gemma", profile="Gemma", mode="router",
                                  running=True, port=8080, host="127.0.0.1",
                                  embeddings=False, reranking=False)
    mp = w._monitored_profile()
    assert mp.mode == "router" and mp.name == "Gemma"
    assert w._poll_api_key(mp) == ak.read_api_key(w.router_base_dir(), "Gemma")
    assert w._poll_api_key(mp) != "sk-wrong"                 # not the stored server key


def test_instance_summary_sends_router_api_key(qtbot, tmp_path, monkeypatch):
    # Regression: the per-row instance summary polled /metrics with NO key, so
    # the auth middleware answered 401 every tick (the recurring router error).
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from llama_launcher.core.instances import Instance
    from llama_launcher.services import api_key as ak
    seen = {}
    monkeypatch.setattr(mw.health, "probe_health",
                        lambda port, timeout=1.0, host="127.0.0.1", **kw: "ready")
    monkeypatch.setattr(mw.metrics, "fetch_metrics",
                        lambda port, timeout=1.0, model=None, api_key=None, host="127.0.0.1", **kw:
                        (seen.update(api_key=api_key) or {}))
    w = mw.MainWindow(); qtbot.addWidget(w)
    ak.ensure_api_key(w.router_base_dir(), "e2b")
    inst = Instance(name="llama-e2b", profile="e2b", mode="router", running=True,
                    port=8080, host="127.0.0.1", embeddings=False, reranking=False)
    w.instance_summary(inst)
    assert seen.get("api_key") == ak.read_api_key(w.router_base_dir(), "e2b")


def test_metrics_report_sends_key_and_model_scope_for_router(qtbot, tmp_path, monkeypatch):
    # Regression: the diagnostic report fetched /metrics with no key/host/scope,
    # so a router with --metrics on always printed "no metrics returned".
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from llama_launcher.services import api_key as ak
    seen = {}
    monkeypatch.setattr(mw.metrics, "fetch_metrics",
        lambda port, timeout=1.0, model=None, api_key=None, host="127.0.0.1", **kw:
        (seen.update(port=port, model=model, api_key=api_key, host=host)
         or {"llamacpp:predicted_tokens_seconds": 5.0}))
    monkeypatch.setattr(mw.metrics, "fetch_slots",
                        lambda port, timeout=1.0, model=None, api_key=None, host="127.0.0.1", **kw: [])
    w = mw.MainWindow(); qtbot.addWidget(w)
    ak.ensure_api_key(w.router_base_dir(), "R")
    w.load_profile(Profile(name="R", image="img", mode="router",
                           settings={"port": 11434, "metrics": True}))
    w._router_statuses = {"m1": "loaded"}          # a loaded model to scope to
    txt = w._metrics_report_text(w.current_profile())
    assert seen["api_key"] == ak.read_api_key(w.router_base_dir(), "R")
    assert seen["model"] == "m1"
    assert seen["port"] == 11434
    assert "generation" in txt
