import subprocess

from llama_launcher.core.router_preset import PresetResult
from llama_launcher.core.spec import Profile, RouterMember, Runtime
from llama_launcher.services import api_key, headless


def _router(name="r", bind="0.0.0.0", port=8080):
    p = Profile(
        name=name,
        image="img",
        runtime=Runtime(bind_host=bind),
        mode="router",
        members=[RouterMember(profile="m1")],
    )
    p.settings["port"] = port
    return p


def _patch_prep(monkeypatch, calls):
    monkeypatch.setattr(
        headless, "resolve_member_pairs", lambda members, base: [("pair",)]
    )
    monkeypatch.setattr(
        headless,
        "render_preset",
        lambda pairs: PresetResult(text="ini", warnings=["w1"]),
    )
    monkeypatch.setattr(
        headless.api_key_store,
        "ensure_api_key",
        lambda base, name: calls.append(("ensure", name)) or "KEY",
    )
    monkeypatch.setattr(
        headless.api_key_store,
        "write_preset",
        lambda base, name, text: calls.append(("preset", text)),
    )
    monkeypatch.setattr(
        headless.api_key_store, "router_dir", lambda base, name: f"/cfg/{name}"
    )
    monkeypatch.setattr(
        headless,
        "build_command",
        lambda p, router_host_dir="": [
            "podman",
            "run",
            "-d",
            "--name",
            f"llama-{p.name}",
            router_host_dir,
        ],
    )


def test_launch_router_success_absent_container(monkeypatch):
    calls = []
    _patch_prep(monkeypatch, calls)
    monkeypatch.setattr(headless, "container_state", lambda name, binary: "absent")
    ran = {}

    def fake_run(argv):
        ran["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(headless, "_run", fake_run)

    res = headless.launch_router(_router(), "/base", "podman")
    assert (
        res.ok and res.name == "llama-r" and res.host == "0.0.0.0" and res.port == 8080
    )
    assert res.warnings == ["w1"] and res.error is None
    assert ran["argv"] == ["podman", "run", "-d", "--name", "llama-r", "/cfg/r"]
    assert ("ensure", "r") in calls and ("preset", "ini") in calls


def test_launch_router_removes_stopped_container_first(monkeypatch):
    _patch_prep(monkeypatch, [])
    monkeypatch.setattr(headless, "container_state", lambda name, binary: "stopped")
    order = []

    def fake_run(argv):
        order.append(argv[1])  # "rm" or "run"
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(headless, "_run", fake_run)

    res = headless.launch_router(_router(), "/base", "podman")
    assert res.ok
    assert order == ["rm", "run"]  # stopped container removed before run


def test_launch_router_run_failure_reports_stderr(monkeypatch):
    _patch_prep(monkeypatch, [])
    monkeypatch.setattr(headless, "container_state", lambda name, binary: "absent")
    monkeypatch.setattr(
        headless,
        "_run",
        lambda argv: subprocess.CompletedProcess(argv, 125, "", "boom\n"),
    )
    res = headless.launch_router(_router(), "/base", "podman")
    assert res.ok is False and res.error == "boom"


def test_stop_router_absent_is_idempotent(monkeypatch):
    monkeypatch.setattr(headless, "container_state", lambda name, binary: "absent")
    called = []
    monkeypatch.setattr(headless, "_run", lambda argv: called.append(argv))
    assert headless.stop_router(_router(), "podman") is True
    assert called == []  # nothing to stop


def test_stop_router_running_success(monkeypatch):
    monkeypatch.setattr(headless, "container_state", lambda name, binary: "running")
    monkeypatch.setattr(
        headless, "_run", lambda argv: subprocess.CompletedProcess(argv, 0, "", "")
    )
    assert headless.stop_router(_router(), "podman") is True


def test_stop_router_failure(monkeypatch):
    monkeypatch.setattr(headless, "container_state", lambda name, binary: "running")
    monkeypatch.setattr(
        headless, "_run", lambda argv: subprocess.CompletedProcess(argv, 1, "", "nope")
    )
    assert headless.stop_router(_router(), "podman") is False


def test_router_status_ready_dials_loopback(monkeypatch):
    seen = {}
    monkeypatch.setattr(headless, "container_state", lambda name, binary: "running")

    def fake_probe(port, host="127.0.0.1", **kw):
        seen["host"] = host
        return "ready"

    monkeypatch.setattr(headless, "probe_health", fake_probe)
    assert headless.router_status(_router(bind="0.0.0.0"), "podman") == "running"
    assert seen["host"] == "127.0.0.1"  # 0.0.0.0 dialed as loopback


def test_router_status_not_running_is_stopped(monkeypatch):
    monkeypatch.setattr(headless, "container_state", lambda name, binary: "absent")
    # probe must not even be called when the container isn't running
    monkeypatch.setattr(
        headless,
        "probe_health",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("probed")),
    )
    assert headless.router_status(_router(), "podman") == "stopped"


def test_wait_ready_returns_true_when_ready(monkeypatch):
    monkeypatch.setattr(
        headless, "probe_health", lambda port, host="127.0.0.1", **k: "ready"
    )
    assert headless.wait_ready("127.0.0.1", 8080, timeout=5) is True


def test_wait_ready_times_out(monkeypatch):
    monkeypatch.setattr(
        headless, "probe_health", lambda port, host="127.0.0.1", **k: "loading"
    )
    ticks = iter([0.0, 0.5, 1.0, 1.5])  # monotonic advances past the deadline
    monkeypatch.setattr(headless.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(headless.time, "sleep", lambda s: None)
    assert headless.wait_ready("127.0.0.1", 8080, timeout=1.0, interval=0.5) is False


def _server(name="s", bind="127.0.0.1", port=8080):
    p = Profile(
        name=name,
        image="img",
        runtime=Runtime(bind_host=bind),
        mode="server",
        model="/models/m.gguf",
    )
    p.settings["port"] = port
    return p


def test_launch_server_success_absent_container(monkeypatch):
    # Server launch must NOT touch the router preset/api-key machinery.
    def boom(*a, **k):
        raise AssertionError("router prep called for a server")

    monkeypatch.setattr(headless.api_key_store, "ensure_api_key", boom)
    monkeypatch.setattr(headless.api_key_store, "write_preset", boom)
    monkeypatch.setattr(headless, "render_preset", boom)

    seen = {}

    def fake_build(p, detach=False):
        seen["detach"] = detach
        return ["podman", "run", "-d", "--name", f"llama-{p.name}"]

    monkeypatch.setattr(headless, "build_command", fake_build)
    monkeypatch.setattr(headless, "container_state", lambda name, binary: "absent")
    ran = {}

    def fake_run(argv):
        ran["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(headless, "_run", fake_run)

    res = headless.launch_server(_server(), "/base", "podman")
    assert (
        res.ok
        and res.name == "llama-s"
        and res.host == "127.0.0.1"
        and res.port == 8080
    )
    assert res.warnings == [] and res.error is None
    assert seen["detach"] is True  # detached argv requested
    assert ran["argv"] == ["podman", "run", "-d", "--name", "llama-s"]


def test_launch_server_removes_stopped_container_first(monkeypatch):
    monkeypatch.setattr(
        headless, "build_command", lambda p, detach=False: ["podman", "run"]
    )
    monkeypatch.setattr(headless, "container_state", lambda name, binary: "stopped")
    order = []
    monkeypatch.setattr(
        headless,
        "_run",
        lambda argv: (
            order.append(argv[1]) or subprocess.CompletedProcess(argv, 0, "", "")
        ),
    )
    res = headless.launch_server(_server(), "/base", "podman")
    assert res.ok
    assert order == ["rm", "run"]  # stopped container removed before run


def test_launch_server_run_failure_reports_stderr(monkeypatch):
    monkeypatch.setattr(
        headless, "build_command", lambda p, detach=False: ["podman", "run"]
    )
    monkeypatch.setattr(headless, "container_state", lambda name, binary: "absent")
    monkeypatch.setattr(
        headless,
        "_run",
        lambda argv: subprocess.CompletedProcess(argv, 125, "", "boom\n"),
    )
    res = headless.launch_server(_server(), "/base", "podman")
    assert res.ok is False and res.error == "boom" and res.warnings == []


def test_launch_dispatches_by_mode(monkeypatch):
    calls = []
    monkeypatch.setattr(
        headless, "launch_router", lambda p, base, binary: calls.append("router") or "R"
    )
    monkeypatch.setattr(
        headless, "launch_server", lambda p, base, binary: calls.append("server") or "S"
    )
    assert headless.launch(_router(), "/base", "podman") == "R"
    assert headless.launch(_server(), "/base", "podman") == "S"
    assert calls == ["router", "server"]


def test_launch_refuses_native_profile():
    p = _server()
    p.runtime.launch_mode = "native"
    res = headless.launch(p, "/base", "podman")
    assert res.ok is False
    assert res.error == "native launch is GUI-only in this version"


def test_launch_refuses_rpc_profile():
    p = _server()
    p.runtime.launch_mode = "rpc"
    res = headless.launch(p, "/base", "podman")
    assert res.ok is False
    assert res.error == "RPC pool launch is GUI-only in this version"


def test_launch_router_serves_the_global_key(tmp_path, monkeypatch):
    # a global key is set; a global-mode router must serve it (per-profile file materialized)
    api_key.write_global_key(tmp_path, "sk-shared")
    captured = {}

    def fake_run(argv):
        captured["argv"] = argv
        import subprocess

        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(headless, "_run", fake_run)
    monkeypatch.setattr(headless, "container_state", lambda *a, **k: "absent")
    monkeypatch.setattr(headless, "resolve_member_pairs", lambda *a, **k: [])
    p = Profile(
        name="R",
        image="img",
        mode="router",
        runtime=Runtime(router_key_mode="global"),
        settings={"port": 8080},
    )
    headless.launch_router(p, tmp_path, "podman")
    assert api_key.read_api_key(tmp_path, "R") == "sk-shared"
