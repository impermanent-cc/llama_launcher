import llama_launcher.app as app
from llama_launcher.core.spec import Profile, Runtime
from llama_launcher.core.validation import Issue
from llama_launcher.services.headless import LaunchResult


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


def _ready_router(monkeypatch):
    _profiles(monkeypatch, [_router("r")], last="r")
    monkeypatch.setattr(app, "binary_available", lambda b: True)
    monkeypatch.setattr(app, "validate", lambda p, **kw: [])


def test_launch_no_wait_prints_started_exit_0(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "launch_router",
                        lambda p, base, binary: LaunchResult(True, "llama-r", "0.0.0.0", 8080, [], None))
    assert app.main(["--launch", "--profile", "r"]) == 0
    assert "started (llama-r) on 0.0.0.0:8080" in capsys.readouterr().out


def test_launch_podman_failure_exit_1(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "launch_router",
                        lambda p, base, binary: LaunchResult(False, "llama-r", "0.0.0.0", 8080, [], "img not found"))
    assert app.main(["--launch", "--profile", "r"]) == 1
    assert "img not found" in capsys.readouterr().err


def test_launch_wait_ready_exit_0(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "launch_router",
                        lambda p, base, binary: LaunchResult(True, "llama-r", "0.0.0.0", 8080, [], None))
    monkeypatch.setattr(app.headless, "wait_ready", lambda host, port, timeout=60.0: True)
    assert app.main(["--launch", "--profile", "r", "--wait"]) == 0
    assert "ready on 0.0.0.0:8080" in capsys.readouterr().out


def test_launch_wait_timeout_exit_5(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "launch_router",
                        lambda p, base, binary: LaunchResult(True, "llama-r", "0.0.0.0", 8080, [], None))
    monkeypatch.setattr(app.headless, "wait_ready", lambda host, port, timeout=60.0: False)
    assert app.main(["--launch", "--profile", "r", "--wait=30"]) == 5
    assert "not ready after 30s" in capsys.readouterr().err


def test_launch_warnings_go_to_stderr(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "launch_router",
                        lambda p, base, binary: LaunchResult(True, "llama-r", "0.0.0.0", 8080, ["dropped m2"], None))
    app.main(["--launch", "--profile", "r"])
    assert "dropped m2" in capsys.readouterr().err


def test_stop_success_exit_0(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "stop_router", lambda p, binary: True)
    assert app.main(["--stop", "--profile", "r"]) == 0
    assert "stopped" in capsys.readouterr().out


def test_stop_failure_exit_1(monkeypatch):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "stop_router", lambda p, binary: False)
    assert app.main(["--stop", "--profile", "r"]) == 1


def test_health_ready_exit_0(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "router_status", lambda p, binary: "running")
    assert app.main(["--health", "--profile", "r"]) == 0
    assert "health: ready" in capsys.readouterr().out


def test_health_loading_exit_3(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "router_status", lambda p, binary: "loading")
    assert app.main(["--health", "--profile", "r"]) == 3
    assert "health: loading" in capsys.readouterr().out


def test_health_stopped_exit_4(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "router_status", lambda p, binary: "stopped")
    assert app.main(["--health", "--profile", "r"]) == 4
    assert "health: stopped" in capsys.readouterr().out
