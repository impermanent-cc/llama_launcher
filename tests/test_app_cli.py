import llama_launcher.app as app
from llama_launcher.core.spec import Profile, Runtime
from llama_launcher.core.validation import Issue


def _profiles(monkeypatch, profiles, last=None):
    by = {p.name: p for p in profiles}
    monkeypatch.setattr(app, "list_profiles", lambda base: list(profiles))
    monkeypatch.setattr(app, "load_config", lambda base: {"last_profile": last})
    monkeypatch.setattr(app, "default_base_dir", lambda: "/base")
    return by


def _router(name="r", mode="router"):
    return Profile(name=name, image="img", runtime=Runtime(), mode=mode)


def test_gate_no_profiles_exits_2(monkeypatch, capsys):
    _profiles(monkeypatch, [])
    assert app.main(["--health", "--profile", "r"]) == 2
    assert "No saved profiles" in capsys.readouterr().err


def test_gate_unknown_profile_exits_2(monkeypatch, capsys):
    _profiles(monkeypatch, [_router("a")])
    assert app.main(["--health", "--profile", "nope"]) == 2
    assert "not found" in capsys.readouterr().err


def test_gate_non_router_exits_2(monkeypatch, capsys):
    _profiles(monkeypatch, [_router("s", mode="server")])
    assert app.main(["--launch", "--profile", "s"]) == 2
    assert "not a router" in capsys.readouterr().err


def test_gate_validation_error_exits_2(monkeypatch, capsys):
    _profiles(monkeypatch, [_router("r")])
    monkeypatch.setattr(app, "binary_available", lambda b: True)
    monkeypatch.setattr(app, "validate", lambda p, **kw: [Issue("error", "bind exposed without key")])
    assert app.main(["--launch", "--profile", "r"]) == 2
    assert "bind exposed" in capsys.readouterr().err


def test_gate_last_profile_fallback(monkeypatch):
    _profiles(monkeypatch, [_router("r")], last="r")
    monkeypatch.setattr(app, "binary_available", lambda b: True)
    monkeypatch.setattr(app, "validate", lambda p, **kw: [])
    monkeypatch.setattr(app.headless, "router_status", lambda p, binary: "running")
    # No --profile given → resolves "r" from last_profile, gate passes, health runs.
    assert app.main(["--health"]) == 0
