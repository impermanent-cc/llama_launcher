import json

import llama_launcher.app as app
import llama_launcher.store.profiles as store_profiles
from llama_launcher.core.spec import Mount, Profile, Runtime, RouterMember
from llama_launcher.core.validation import Issue
from llama_launcher.services.headless import LaunchResult


def _profiles(monkeypatch, profiles, last=None):
    by = {p.name: p for p in profiles}
    monkeypatch.setattr(app, "list_profiles", lambda base: list(profiles))
    monkeypatch.setattr(app, "load_config", lambda base: {"last_profile": last})
    monkeypatch.setattr(app, "default_base_dir", lambda: "/base")
    # base_dir here is the fake string "/base", not a real Path, so the real
    # resolve_member_pairs (which stats the filesystem via list_profiles)
    # would blow up. Tests that care about real member resolution re-patch
    # this to store_profiles.resolve_member_pairs after calling _profiles().
    monkeypatch.setattr(app, "resolve_member_pairs", lambda members, base: [])
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


def _server(name="s", bind="127.0.0.1", **settings):
    return Profile(
        name=name, image="img", runtime=Runtime(bind_host=bind), mode="server",
        mounts=[Mount(host="/host/models", container="/models", role="model")],
        model="/models/m.gguf", settings={"port": 8080, **settings},
    )


def test_gate_valid_server_passes_and_launches_exit_0(monkeypatch, capsys):
    # Real validate(): loopback bind + model-under-mount = no errors, gate passes.
    _profiles(monkeypatch, [_server("s")], last="s")
    monkeypatch.setattr(app, "binary_available", lambda b: True)
    monkeypatch.setattr(app.headless, "launch",
                        lambda p, base, binary: LaunchResult(True, "llama-s", "127.0.0.1", 8080, [], None))
    assert app.main(["--launch", "--profile", "s"]) == 0
    assert "started (llama-s) on 127.0.0.1:8080" in capsys.readouterr().out


def test_gate_exposed_keyless_server_exits_2(monkeypatch, capsys):
    # bind_host past loopback + no api-key setting → real validate() refuses it.
    _profiles(monkeypatch, [_server("s", bind="0.0.0.0")], last="s")
    monkeypatch.setattr(app, "binary_available", lambda b: True)
    assert app.main(["--launch", "--profile", "s"]) == 2
    assert "without an API key" in capsys.readouterr().err


def test_launch_router_still_works_via_dispatcher(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "launch",
                        lambda p, base, binary: LaunchResult(True, "llama-r", "0.0.0.0", 8080, [], None))
    assert app.main(["--launch", "--profile", "r"]) == 0
    assert "started (llama-r) on 0.0.0.0:8080" in capsys.readouterr().out


def test_gate_validation_error_exits_2(monkeypatch, capsys):
    _profiles(monkeypatch, [_router("r")])
    monkeypatch.setattr(app, "binary_available", lambda b: True)
    monkeypatch.setattr(app, "validate", lambda p, **kw: [Issue("error", "bind exposed without key")])
    assert app.main(["--launch", "--profile", "r"]) == 2
    assert "bind exposed" in capsys.readouterr().err


def test_gate_router_with_valid_member_passes_real_validate(monkeypatch, capsys):
    """Regression for the members= bug: a real router with a resolvable member
    and no exposure problem (loopback bind_host) must pass the REAL validate(),
    not just a monkeypatched stub. Isolates the members rule specifically: the
    only way this profile could fail validate() is if members isn't threaded
    through to _validate_router."""
    router = Profile(
        name="r", image="img", runtime=Runtime(), mode="router",
        mounts=[Mount(host="/host/models", container="/models", role="model")],
        members=[RouterMember(profile="m1")],
    )
    member = Profile(name="m1", image="img2", runtime=Runtime(), model="/models/model.gguf")
    _profiles(monkeypatch, [router, member], last="r")
    # Use the REAL resolve_member_pairs (not the "/base"-safe stub _profiles()
    # installs by default) so this test exercises the actual member-resolution
    # path the members= bug fix depends on. It resolves via
    # store.profiles.list_profiles internally, not app.list_profiles, so that
    # is what needs mocking here.
    monkeypatch.setattr(store_profiles, "list_profiles", lambda base: [router, member])
    monkeypatch.setattr(app, "resolve_member_pairs", store_profiles.resolve_member_pairs)
    monkeypatch.setattr(app, "binary_available", lambda b: True)
    monkeypatch.setattr(app.headless, "router_status", lambda p, binary: "running")
    assert app.main(["--health", "--profile", "r"]) == 0
    assert "health: ready" in capsys.readouterr().out


def test_gate_router_without_members_exits_2_real_validate(monkeypatch, capsys):
    """Legitimate case of the "needs at least one model" error: a router with
    NO members must still be refused by the real validate(), so the members
    fix doesn't over-correct into accepting empty routers."""
    router = Profile(name="r", image="img", runtime=Runtime(), mode="router")
    # _profiles() installs a stub resolve_member_pairs returning [] by default
    # (see its docstring), which is exactly the "no members" case this test
    # wants: it exercises real validate()/_validate_router() with an empty
    # members list, same as the real resolve_member_pairs would produce for a
    # router with no RouterMember entries.
    _profiles(monkeypatch, [router], last="r")
    monkeypatch.setattr(app, "binary_available", lambda b: True)
    assert app.main(["--health", "--profile", "r"]) == 2
    assert "at least one model" in capsys.readouterr().err


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
    monkeypatch.setattr(app.headless, "launch",
                        lambda p, base, binary: LaunchResult(True, "llama-r", "0.0.0.0", 8080, [], None))
    assert app.main(["--launch", "--profile", "r"]) == 0
    assert "started (llama-r) on 0.0.0.0:8080" in capsys.readouterr().out


def test_launch_podman_failure_exit_1(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "launch",
                        lambda p, base, binary: LaunchResult(False, "llama-r", "0.0.0.0", 8080, [], "img not found"))
    assert app.main(["--launch", "--profile", "r"]) == 1
    assert "img not found" in capsys.readouterr().err


def test_launch_wait_ready_exit_0(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "launch",
                        lambda p, base, binary: LaunchResult(True, "llama-r", "0.0.0.0", 8080, [], None))
    monkeypatch.setattr(app.headless, "wait_ready", lambda host, port, timeout=60.0: True)
    assert app.main(["--launch", "--profile", "r", "--wait"]) == 0
    assert "ready on 0.0.0.0:8080" in capsys.readouterr().out


def test_launch_wait_timeout_exit_5(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "launch",
                        lambda p, base, binary: LaunchResult(True, "llama-r", "0.0.0.0", 8080, [], None))
    monkeypatch.setattr(app.headless, "wait_ready", lambda host, port, timeout=60.0: False)
    assert app.main(["--launch", "--profile", "r", "--wait=30"]) == 5
    assert "not ready after 30s" in capsys.readouterr().err


def test_launch_warnings_go_to_stderr(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "launch",
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


def test_json_gate_unknown_profile_one_object_exit_2(monkeypatch, capsys):
    _profiles(monkeypatch, [_router("a")])
    assert app.main(["--health", "--profile", "nope", "--json"]) == 2
    cap = capsys.readouterr()
    obj = json.loads(cap.out)               # exactly one JSON object on stdout
    assert obj["action"] == "health"
    assert obj["ok"] is False
    assert obj["status"] is None
    assert "not found" in obj["error"]
    assert obj["name"] == "nope"
    assert obj["warnings"] == []
    assert cap.err == ""                     # nothing on stderr in JSON mode


def test_json_gate_no_profiles_exit_2(monkeypatch, capsys):
    _profiles(monkeypatch, [])
    assert app.main(["--launch", "--profile", "r", "--json"]) == 2
    cap = capsys.readouterr()
    obj = json.loads(cap.out)
    assert obj["action"] == "launch" and obj["ok"] is False and obj["error"]
    assert cap.err == ""


def test_text_gate_refusal_unchanged(monkeypatch, capsys):
    _profiles(monkeypatch, [_router("a")])
    assert app.main(["--health", "--profile", "nope"]) == 2
    cap = capsys.readouterr()
    assert "not found" in cap.err            # text mode: message still on stderr
    assert cap.out == ""                      # text mode: nothing on stdout


def test_json_launch_no_wait_started(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "launch",
                        lambda p, base, binary: LaunchResult(True, "llama-r", "0.0.0.0", 8080, [], None))
    assert app.main(["--launch", "--profile", "r", "--json"]) == 0
    cap = capsys.readouterr()
    obj = json.loads(cap.out)
    assert obj == {"action": "launch", "ok": True, "status": "started",
                   "name": "llama-r", "host": "0.0.0.0", "port": 8080,
                   "warnings": [], "error": None}
    assert cap.err == ""


def test_json_launch_with_warning(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "launch",
                        lambda p, base, binary: LaunchResult(True, "llama-r", "0.0.0.0", 8080, ["dropped m2"], None))
    assert app.main(["--launch", "--profile", "r", "--json"]) == 0
    cap = capsys.readouterr()
    obj = json.loads(cap.out)
    assert obj["warnings"] == ["dropped m2"]   # warning is INSIDE the object
    assert cap.err == ""                        # not on stderr in JSON mode


def test_json_launch_run_failed(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "launch",
                        lambda p, base, binary: LaunchResult(False, "llama-r", "0.0.0.0", 8080, [], "img not found"))
    assert app.main(["--launch", "--profile", "r", "--json"]) == 1
    obj = json.loads(capsys.readouterr().out)
    assert obj["ok"] is False and obj["status"] is None and obj["error"] == "img not found"


def test_json_launch_wait_timeout(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "launch",
                        lambda p, base, binary: LaunchResult(True, "llama-r", "0.0.0.0", 8080, [], None))
    monkeypatch.setattr(app.headless, "wait_ready", lambda host, port, timeout=60.0: False)
    assert app.main(["--launch", "--profile", "r", "--wait=30", "--json"]) == 5
    obj = json.loads(capsys.readouterr().out)
    assert obj["ok"] is False and obj["status"] == "started"
    assert "30s" in obj["error"]                # error mentions the timeout


def test_json_launch_wait_ready(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "launch",
                        lambda p, base, binary: LaunchResult(True, "llama-r", "0.0.0.0", 8080, [], None))
    monkeypatch.setattr(app.headless, "wait_ready", lambda host, port, timeout=60.0: True)
    assert app.main(["--launch", "--profile", "r", "--wait", "--json"]) == 0
    obj = json.loads(capsys.readouterr().out)
    assert obj["status"] == "ready" and obj["ok"] is True


def test_json_stop_success(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "stop_router", lambda p, binary: True)
    assert app.main(["--stop", "--profile", "r", "--json"]) == 0
    cap = capsys.readouterr()
    obj = json.loads(cap.out)
    assert obj["action"] == "stop" and obj["ok"] is True and obj["status"] == "stopped"
    assert cap.err == ""


def test_json_stop_failure(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "stop_router", lambda p, binary: False)
    assert app.main(["--stop", "--profile", "r", "--json"]) == 1
    obj = json.loads(capsys.readouterr().out)
    assert obj["ok"] is False and obj["status"] is None


def test_json_health_ready(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "router_status", lambda p, binary: "running")
    assert app.main(["--health", "--profile", "r", "--json"]) == 0
    obj = json.loads(capsys.readouterr().out)
    assert obj["ok"] is True and obj["status"] == "ready"


def test_json_health_loading(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "router_status", lambda p, binary: "loading")
    assert app.main(["--health", "--profile", "r", "--json"]) == 3
    obj = json.loads(capsys.readouterr().out)
    assert obj["ok"] is False and obj["status"] == "loading"


def test_json_health_down(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "router_status", lambda p, binary: "stopped")
    assert app.main(["--health", "--profile", "r", "--json"]) == 4
    obj = json.loads(capsys.readouterr().out)
    assert obj["ok"] is False and obj["status"] == "stopped"
