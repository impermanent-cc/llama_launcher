import dataclasses
import json

import llama_launcher.app as app
import llama_launcher.store.profiles as store_profiles
from llama_launcher.core.spec import Mount, Profile, RouterMember, Runtime
from llama_launcher.core.validation import Issue
from llama_launcher.services import api_key
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
        name=name,
        image="img",
        runtime=Runtime(bind_host=bind),
        mode="server",
        mounts=[Mount(host="/host/models", container="/models", role="model")],
        model="/models/m.gguf",
        settings={"port": 8080, **settings},
    )


def test_gate_valid_server_passes_and_launches_exit_0(monkeypatch, capsys):
    # Real validate(): loopback bind + model-under-mount = no errors, gate passes.
    _profiles(monkeypatch, [_server("s")], last="s")
    monkeypatch.setattr(app, "binary_available", lambda b: True)
    monkeypatch.setattr(
        app.headless,
        "launch",
        lambda p, base, binary: LaunchResult(
            True, "llama-s", "127.0.0.1", 8080, [], None
        ),
    )
    assert app.main(["--launch", "--profile", "s"]) == 0
    assert "started (llama-s) on 127.0.0.1:8080" in capsys.readouterr().out


def test_gate_exposed_keyless_server_exits_2(monkeypatch, capsys):
    # bind_host past loopback + no api-key setting means real validate() refuses it.
    _profiles(monkeypatch, [_server("s", bind="0.0.0.0")], last="s")
    monkeypatch.setattr(app, "binary_available", lambda b: True)
    assert app.main(["--launch", "--profile", "s"]) == 2
    assert "without an API key" in capsys.readouterr().err


def test_launch_router_still_works_via_dispatcher(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(
        app.headless,
        "launch",
        lambda p, base, binary: LaunchResult(
            True, "llama-r", "0.0.0.0", 8080, [], None
        ),
    )
    assert app.main(["--launch", "--profile", "r"]) == 0
    assert "started (llama-r) on 0.0.0.0:8080" in capsys.readouterr().out


def test_gate_validation_error_exits_2(monkeypatch, capsys):
    _profiles(monkeypatch, [_router("r")])
    monkeypatch.setattr(app, "binary_available", lambda b: True)
    monkeypatch.setattr(
        app, "validate", lambda p, **kw: [Issue("error", "bind exposed without key")]
    )
    assert app.main(["--launch", "--profile", "r"]) == 2
    assert "bind exposed" in capsys.readouterr().err


def test_gate_router_with_valid_member_passes_real_validate(monkeypatch, capsys):
    """A real router with a resolvable member and no exposure problem (loopback
    bind_host) passes the REAL validate(), not just a monkeypatched stub. This
    isolates the members rule: the only way this profile can fail validate()
    is if members isn't threaded through to _validate_router."""
    router = Profile(
        name="r",
        image="img",
        runtime=Runtime(),
        mode="router",
        mounts=[Mount(host="/host/models", container="/models", role="model")],
        members=[RouterMember(profile="m1")],
    )
    member = Profile(
        name="m1", image="img2", runtime=Runtime(), model="/models/model.gguf"
    )
    _profiles(monkeypatch, [router, member], last="r")
    # Use the REAL resolve_member_pairs (not the "/base"-safe stub _profiles()
    # installs by default) so this test exercises the actual member-resolution
    # path. It resolves via store.profiles.list_profiles internally, not
    # app.list_profiles, so that is what needs mocking here.
    monkeypatch.setattr(store_profiles, "list_profiles", lambda base: [router, member])
    monkeypatch.setattr(
        app, "resolve_member_pairs", store_profiles.resolve_member_pairs
    )
    monkeypatch.setattr(app, "binary_available", lambda b: True)
    monkeypatch.setattr(app.headless, "router_status", lambda p, binary: "running")
    assert app.main(["--health", "--profile", "r"]) == 0
    assert "health: ready" in capsys.readouterr().out


def test_gate_router_without_members_exits_2_real_validate(monkeypatch, capsys):
    """Legitimate case of the "needs at least one model" error: a router with
    NO members is refused by the real validate()."""
    router = Profile(name="r", image="img", runtime=Runtime(), mode="router")
    # _profiles() installs a stub resolve_member_pairs returning [] by default,
    # which is exactly the "no members" case this test
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
    # No --profile given means resolves "r" from last_profile, gate passes, health runs.
    assert app.main(["--health"]) == 0


def _ready_router(monkeypatch):
    _profiles(monkeypatch, [_router("r")], last="r")
    monkeypatch.setattr(app, "binary_available", lambda b: True)
    monkeypatch.setattr(app, "validate", lambda p, **kw: [])


def test_launch_no_wait_prints_started_exit_0(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(
        app.headless,
        "launch",
        lambda p, base, binary: LaunchResult(
            True, "llama-r", "0.0.0.0", 8080, [], None
        ),
    )
    assert app.main(["--launch", "--profile", "r"]) == 0
    assert "started (llama-r) on 0.0.0.0:8080" in capsys.readouterr().out


def test_launch_podman_failure_exit_1(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(
        app.headless,
        "launch",
        lambda p, base, binary: LaunchResult(
            False, "llama-r", "0.0.0.0", 8080, [], "img not found"
        ),
    )
    assert app.main(["--launch", "--profile", "r"]) == 1
    assert "img not found" in capsys.readouterr().err


def test_launch_wait_ready_exit_0(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(
        app.headless,
        "launch",
        lambda p, base, binary: LaunchResult(
            True, "llama-r", "0.0.0.0", 8080, [], None
        ),
    )
    monkeypatch.setattr(
        app.headless, "wait_ready", lambda host, port, timeout=60.0: True
    )
    assert app.main(["--launch", "--profile", "r", "--wait"]) == 0
    assert "ready on 0.0.0.0:8080" in capsys.readouterr().out


def test_launch_wait_timeout_exit_5(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(
        app.headless,
        "launch",
        lambda p, base, binary: LaunchResult(
            True, "llama-r", "0.0.0.0", 8080, [], None
        ),
    )
    monkeypatch.setattr(
        app.headless, "wait_ready", lambda host, port, timeout=60.0: False
    )
    assert app.main(["--launch", "--profile", "r", "--wait=30"]) == 5
    assert "not ready after 30s" in capsys.readouterr().err


def test_launch_warnings_go_to_stderr(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(
        app.headless,
        "launch",
        lambda p, base, binary: LaunchResult(
            True, "llama-r", "0.0.0.0", 8080, ["dropped m2"], None
        ),
    )
    app.main(["--launch", "--profile", "r"])
    assert "dropped m2" in capsys.readouterr().err


def test_launch_native_profile_refused_by_cli(monkeypatch, capsys):
    native = _server("s")
    native.runtime.launch_mode = "native"
    native.runtime.native_binary = "/opt/bin/llama-server"
    _profiles(monkeypatch, [native], last="s")
    monkeypatch.setattr(app, "binary_available", lambda b: True)
    monkeypatch.setattr(app, "validate", lambda p, **kw: [])
    assert app.main(["--launch", "--profile", "s"]) == 1
    assert "gui-only" in capsys.readouterr().err.lower()


def test_stop_native_profile_refused_by_cli(monkeypatch, capsys):
    native = _server("s")
    native.runtime.launch_mode = "native"
    native.runtime.native_binary = "/opt/bin/llama-server"
    _profiles(monkeypatch, [native], last="s")
    monkeypatch.setattr(app, "binary_available", lambda b: True)
    monkeypatch.setattr(app, "validate", lambda p, **kw: [])
    # must not touch the container stop path
    monkeypatch.setattr(
        app.headless,
        "stop_router",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("container path taken")),
    )
    assert app.main(["--stop", "--profile", "s"]) == 1
    assert "gui-only" in capsys.readouterr().err.lower()


def test_health_native_profile_refused_by_cli(monkeypatch, capsys):
    native = _server("s")
    native.runtime.launch_mode = "native"
    native.runtime.native_binary = "/opt/bin/llama-server"
    _profiles(monkeypatch, [native], last="s")
    monkeypatch.setattr(app, "binary_available", lambda b: True)
    monkeypatch.setattr(app, "validate", lambda p, **kw: [])
    monkeypatch.setattr(
        app.headless,
        "router_status",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("container path taken")),
    )
    assert app.main(["--health", "--profile", "s"]) == 1
    assert "gui-only" in capsys.readouterr().err.lower()


def test_stop_success_exit_0(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "stop_router", lambda p, binary, timeout=10: True)
    assert app.main(["--stop", "--profile", "r"]) == 0
    assert "stopped" in capsys.readouterr().out


def test_stop_failure_exit_1(monkeypatch):
    _ready_router(monkeypatch)
    monkeypatch.setattr(
        app.headless, "stop_router", lambda p, binary, timeout=10: False
    )
    assert app.main(["--stop", "--profile", "r"]) == 1


def test_stop_honors_profile_stop_timeout(monkeypatch):
    """The CLI --stop threads the profile's configurable grace period into the
    container teardown, matching the GUI Stop button."""
    slow = Profile(
        name="r", image="img", mode="router", runtime=Runtime(stop_timeout=50)
    )
    _profiles(monkeypatch, [slow], last="r")
    monkeypatch.setattr(app, "binary_available", lambda b: True)
    monkeypatch.setattr(app, "validate", lambda p, **kw: [])
    seen = {}
    monkeypatch.setattr(
        app.headless,
        "stop_router",
        lambda p, binary, timeout=10: seen.setdefault("timeout", timeout) or True,
    )
    app.main(["--stop", "--profile", "r"])
    assert seen["timeout"] == 50


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
    obj = json.loads(cap.out)  # exactly one JSON object on stdout
    assert obj["action"] == "health"
    assert obj["ok"] is False
    assert obj["status"] is None
    assert "not found" in obj["error"]
    assert obj["name"] == "nope"
    assert obj["warnings"] == []
    assert cap.err == ""  # nothing on stderr in JSON mode


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
    assert "not found" in cap.err  # text mode: message still on stderr
    assert cap.out == ""  # text mode: nothing on stdout


def test_json_launch_no_wait_started(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(
        app.headless,
        "launch",
        lambda p, base, binary: LaunchResult(
            True, "llama-r", "0.0.0.0", 8080, [], None
        ),
    )
    assert app.main(["--launch", "--profile", "r", "--json"]) == 0
    cap = capsys.readouterr()
    obj = json.loads(cap.out)
    assert obj == {
        "action": "launch",
        "ok": True,
        "status": "started",
        "name": "llama-r",
        "host": "0.0.0.0",
        "port": 8080,
        "warnings": [],
        "error": None,
    }
    assert cap.err == ""


def test_json_launch_with_warning(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(
        app.headless,
        "launch",
        lambda p, base, binary: LaunchResult(
            True, "llama-r", "0.0.0.0", 8080, ["dropped m2"], None
        ),
    )
    assert app.main(["--launch", "--profile", "r", "--json"]) == 0
    cap = capsys.readouterr()
    obj = json.loads(cap.out)
    assert obj["warnings"] == ["dropped m2"]  # warning is INSIDE the object
    assert cap.err == ""  # not on stderr in JSON mode


def test_json_launch_run_failed(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(
        app.headless,
        "launch",
        lambda p, base, binary: LaunchResult(
            False, "llama-r", "0.0.0.0", 8080, [], "img not found"
        ),
    )
    assert app.main(["--launch", "--profile", "r", "--json"]) == 1
    obj = json.loads(capsys.readouterr().out)
    assert (
        obj["ok"] is False and obj["status"] is None and obj["error"] == "img not found"
    )


def test_json_launch_wait_timeout(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(
        app.headless,
        "launch",
        lambda p, base, binary: LaunchResult(
            True, "llama-r", "0.0.0.0", 8080, [], None
        ),
    )
    monkeypatch.setattr(
        app.headless, "wait_ready", lambda host, port, timeout=60.0: False
    )
    assert app.main(["--launch", "--profile", "r", "--wait=30", "--json"]) == 5
    obj = json.loads(capsys.readouterr().out)
    assert obj["ok"] is False and obj["status"] == "started"
    assert "30s" in obj["error"]  # error mentions the timeout


def test_json_launch_wait_ready(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(
        app.headless,
        "launch",
        lambda p, base, binary: LaunchResult(
            True, "llama-r", "0.0.0.0", 8080, [], None
        ),
    )
    monkeypatch.setattr(
        app.headless, "wait_ready", lambda host, port, timeout=60.0: True
    )
    assert app.main(["--launch", "--profile", "r", "--wait", "--json"]) == 0
    obj = json.loads(capsys.readouterr().out)
    assert obj["status"] == "ready" and obj["ok"] is True


def test_json_stop_success(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(app.headless, "stop_router", lambda p, binary, timeout=10: True)
    assert app.main(["--stop", "--profile", "r", "--json"]) == 0
    cap = capsys.readouterr()
    obj = json.loads(cap.out)
    assert obj["action"] == "stop" and obj["ok"] is True and obj["status"] == "stopped"
    assert cap.err == ""


def test_json_stop_failure(monkeypatch, capsys):
    _ready_router(monkeypatch)
    monkeypatch.setattr(
        app.headless, "stop_router", lambda p, binary, timeout=10: False
    )
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


def _estimate_env(monkeypatch, free_mib=(30000,), ram=64 * 1024**3):
    from llama_launcher.core.gguf import GgufMeta, TensorInfo
    from llama_launcher.services import gpu, model_info, pool_preflight

    ts = [TensorInfo("token_embd.weight", 1, 0, 1024**2)]
    for i in range(4):
        ts.append(TensorInfo(f"blk.{i}.attn_q.weight", 1, 0, 256 * 1024**2))
        ts.append(TensorInfo(f"blk.{i}.ffn_up.weight", 1, 0, 256 * 1024**2))
    ts.append(TensorInfo("output.weight", 1, 0, 1024**2))
    meta = GgufMeta(
        arch="llama",
        n_layers=4,
        n_head=8,
        n_head_kv=8,
        n_embd=64,
        ctx_train=4096,
        n_ff=256,
        n_vocab=1000,
        tensors=tuple(ts),
    )
    monkeypatch.setattr(model_info, "read_model", lambda host: (meta, 2 * 1024**3))
    monkeypatch.setattr(model_info, "inspect_file", lambda path, mounts: (None, 0))
    monkeypatch.setattr(model_info, "sibling_ggufs", lambda host: [])
    from llama_launcher.services.gpu import GpuStat

    monkeypatch.setattr(
        gpu,
        "query_gpus",
        lambda ssh_target="": [GpuStat("g", 0, f, f, 0, 40) for f in free_mib],
    )
    monkeypatch.setattr(pool_preflight, "free_ram_bytes", lambda ssh_target="": ram)
    monkeypatch.setattr(app, "gpu_ssh_target", lambda base, node: "")
    monkeypatch.setattr(app, "binary_available", lambda b: True)


def test_estimate_prints_lines_and_exits_0(monkeypatch, capsys):
    _profiles(monkeypatch, [_server("s")])
    _estimate_env(monkeypatch)
    assert app.main(["--estimate", "--profile", "s"]) == 0
    out = capsys.readouterr().out
    assert "GPU0" in out and "RAM" in out
    assert "<span" not in out
    assert out.isascii()


def test_estimate_json(monkeypatch, capsys):
    _profiles(monkeypatch, [_server("s")])
    _estimate_env(monkeypatch)
    assert app.main(["--estimate", "--profile", "s", "--json"]) == 0
    obj = json.loads(capsys.readouterr().out)
    assert obj["action"] == "estimate" and obj["ok"] is True
    assert obj["estimate"]["cards"][0]["fits"] is True
    assert obj["status"] is None and obj["host"] is None and obj["port"] is None
    assert obj["warnings"] == [] and obj["error"] is None


def test_estimate_json_carries_the_balanced_split(monkeypatch, capsys):
    """A two-card --estimate --json readout carries balanced_split with a
    value string that matches its own layers_per_card, one boundary between
    the two cards and a bool fits flag."""
    _profiles(monkeypatch, [_server("s")])
    _estimate_env(monkeypatch, free_mib=(30000, 12000))
    assert app.main(["--estimate", "--profile", "s", "--json"]) == 0
    obj = json.loads(capsys.readouterr().out)
    b = obj["estimate"]["balanced_split"]
    assert set(b) == {"value", "layers_per_card", "boundary_layers", "fits"}
    assert b["value"] == ",".join(str(n) for n in b["layers_per_card"])
    assert len(b["layers_per_card"]) == 2
    assert len(b["boundary_layers"]) == 1
    assert b["fits"] is True


def test_estimate_json_single_card_has_no_balanced_split(monkeypatch, capsys):
    """A one-card --estimate --json readout has no split to balance, so the
    balanced_split key is absent from the estimate object."""
    _profiles(monkeypatch, [_server("s")])
    _estimate_env(monkeypatch, free_mib=(30000,))
    assert app.main(["--estimate", "--profile", "s", "--json"]) == 0
    obj = json.loads(capsys.readouterr().out)
    assert "balanced_split" not in obj["estimate"]


def test_estimate_exit_3_when_over(monkeypatch, capsys):
    _profiles(monkeypatch, [_server("s", **{"fit": "off"})])
    _estimate_env(monkeypatch, free_mib=(512,))
    assert app.main(["--estimate", "--profile", "s"]) == 3
    out = capsys.readouterr().out
    assert "exceeds" in out
    assert "<span" not in out
    assert "&gt;" not in out
    assert ">" in out


def test_estimate_exits_6_when_only_ram_is_over_budget(monkeypatch, capsys):
    _profiles(monkeypatch, [_server("s")])
    _estimate_env(monkeypatch, free_mib=(30000,), ram=1)
    assert app.main(["--estimate", "--profile", "s", "--json"]) == 6
    obj = json.loads(capsys.readouterr().out)
    assert obj["ok"] is True
    assert obj["estimate"]["ram"]["fits"] is False


def test_estimate_exits_3_when_a_card_and_ram_are_over(monkeypatch):
    _profiles(monkeypatch, [_server("s", **{"fit": "off"})])
    _estimate_env(monkeypatch, free_mib=(512,), ram=1)
    assert app.main(["--estimate", "--profile", "s"]) == 3


def test_estimate_exits_0_when_ram_is_unknown(monkeypatch):
    _profiles(monkeypatch, [_server("s")])
    _estimate_env(monkeypatch, free_mib=(30000,), ram=None)
    assert app.main(["--estimate", "--profile", "s"]) == 0


def test_estimate_names_an_unmounted_draft(monkeypatch, capsys):
    p = dataclasses.replace(_server("s"), draft_model="/nowhere/d.gguf")
    _profiles(monkeypatch, [p])
    _estimate_env(monkeypatch, free_mib=(30000,))
    app.main(["--estimate", "--profile", "s"])
    out = capsys.readouterr().out
    assert "Draft model /nowhere/d.gguf lies under no configured folder" in out


def test_estimate_does_not_name_a_mounted_draft(monkeypatch, capsys):
    p = dataclasses.replace(_server("s"), draft_model="/models/d.gguf")
    _profiles(monkeypatch, [p])
    _estimate_env(monkeypatch, free_mib=(30000,))
    app.main(["--estimate", "--profile", "s"])
    out = capsys.readouterr().out
    assert "lies under no configured folder" not in out


def test_estimate_exit_2_without_gpus(monkeypatch, capsys):
    _profiles(monkeypatch, [_server("s")])
    _estimate_env(monkeypatch, free_mib=())
    assert app.main(["--estimate", "--profile", "s"]) == 2
    assert "no GPU" in capsys.readouterr().err


def test_estimate_router_without_own_model_exits_2(monkeypatch, capsys):
    _ready_router(monkeypatch)
    assert app.main(["--estimate", "--profile", "r"]) == 2
    assert "no model of its own" in capsys.readouterr().err


def test_estimate_json_draft_increases_card_estimate(monkeypatch, capsys):
    from llama_launcher.core.gguf import GgufMeta, TensorInfo
    from llama_launcher.services import model_info

    draft_ts = [TensorInfo("token_embd.weight", 1, 0, 1024**2)]
    for i in range(2):
        draft_ts.append(TensorInfo(f"blk.{i}.attn_q.weight", 1, 0, 64 * 1024**2))
    draft_meta = GgufMeta(
        arch="llama",
        n_layers=2,
        n_head=8,
        n_head_kv=8,
        n_embd=64,
        ctx_train=4096,
        n_ff=256,
        n_vocab=1000,
        tensors=tuple(draft_ts),
    )
    profile = dataclasses.replace(_server("s"), draft_model="/models/draft.gguf")
    _profiles(monkeypatch, [profile])
    _estimate_env(monkeypatch)
    assert app.main(["--estimate", "--profile", "s", "--json"]) == 0
    baseline = json.loads(capsys.readouterr().out)["estimate"]["cards"][0]["est"]

    monkeypatch.setattr(
        model_info,
        "inspect_file",
        lambda path, mounts: (
            (draft_meta, 128 * 1024**2) if path == "/models/draft.gguf" else (None, 0)
        ),
    )
    assert app.main(["--estimate", "--profile", "s", "--json"]) == 0
    with_draft = json.loads(capsys.readouterr().out)["estimate"]["cards"][0]["est"]
    assert with_draft > baseline


def test_gate_treats_global_key_as_present_for_health(tmp_path, monkeypatch):
    # This test isolates the api-key exposure check; the container runtime binary
    # (a shutil.which PATH probe) is incidental, so treat it as present -- a
    # headless CI container has no podman, which would otherwise inject an
    # unrelated "Runtime 'podman' not found on PATH" gate error.
    monkeypatch.setattr(app, "binary_available", lambda binary: True)
    # global-mode router, only a global key exists, never launched (no per-profile file)
    api_key.write_global_key(tmp_path, "sk-shared")
    # A resolvable member is included so the real validate()'s "needs at least
    # one model" rule (unrelated to this test) doesn't also fire; this test
    # isolates the api-key exposure check.
    store_profiles.save_profile(
        Profile(
            name="R",
            image="img",
            mode="router",
            runtime=Runtime(bind_host="0.0.0.0", router_key_mode="global"),
            mounts=[Mount(host="/host/models", container="/models", role="model")],
            members=[RouterMember(profile="m1")],
            settings={"port": 8080},
        ),
        tmp_path,
    )
    store_profiles.save_profile(
        Profile(name="m1", image="img2", runtime=Runtime(), model="/models/model.gguf"),
        tmp_path,
    )
    _p, code, msg = app._resolve_and_gate("health", "R", tmp_path)
    # bound to 0.0.0.0 => exposure rule fires ONLY if the key looks absent;
    # with a global key present there must be no error.
    assert code is None, msg
