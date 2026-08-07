import subprocess
from llama_launcher.core.spec import Profile, RouterMember, Runtime
from llama_launcher.services import headless
from llama_launcher.core.router_preset import PresetResult


def _router(name="r", bind="0.0.0.0", port=8080):
    p = Profile(name=name, image="img", runtime=Runtime(bind_host=bind), mode="router",
                members=[RouterMember(profile="m1")])
    p.settings["port"] = port
    return p


def _patch_prep(monkeypatch, calls):
    monkeypatch.setattr(headless, "resolve_member_pairs",
                        lambda members, base: [("pair",)])
    monkeypatch.setattr(headless, "render_preset",
                        lambda pairs: PresetResult(text="ini", warnings=["w1"]))
    monkeypatch.setattr(headless.api_key_store, "ensure_api_key",
                        lambda base, name: calls.append(("ensure", name)) or "KEY")
    monkeypatch.setattr(headless.api_key_store, "write_preset",
                        lambda base, name, text: calls.append(("preset", text)))
    monkeypatch.setattr(headless.api_key_store, "router_dir",
                        lambda base, name: f"/cfg/{name}")
    monkeypatch.setattr(headless, "build_command",
                        lambda p, router_host_dir="": ["podman", "run", "-d",
                                                       "--name", f"llama-{p.name}", router_host_dir])


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
    assert res.ok and res.name == "llama-r" and res.host == "0.0.0.0" and res.port == 8080
    assert res.warnings == ["w1"] and res.error is None
    assert ran["argv"] == ["podman", "run", "-d", "--name", "llama-r", "/cfg/r"]
    assert ("ensure", "r") in calls and ("preset", "ini") in calls


def test_launch_router_removes_stopped_container_first(monkeypatch):
    _patch_prep(monkeypatch, [])
    monkeypatch.setattr(headless, "container_state", lambda name, binary: "stopped")
    order = []
    def fake_run(argv):
        order.append(argv[1])   # "rm" or "run"
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(headless, "_run", fake_run)

    res = headless.launch_router(_router(), "/base", "podman")
    assert res.ok
    assert order == ["rm", "run"]          # stopped container removed before run


def test_launch_router_run_failure_reports_stderr(monkeypatch):
    _patch_prep(monkeypatch, [])
    monkeypatch.setattr(headless, "container_state", lambda name, binary: "absent")
    monkeypatch.setattr(headless, "_run",
                        lambda argv: subprocess.CompletedProcess(argv, 125, "", "boom\n"))
    res = headless.launch_router(_router(), "/base", "podman")
    assert res.ok is False and res.error == "boom"


def test_stop_router_absent_is_idempotent(monkeypatch):
    monkeypatch.setattr(headless, "container_state", lambda name, binary: "absent")
    called = []
    monkeypatch.setattr(headless, "_run", lambda argv: called.append(argv))
    assert headless.stop_router(_router(), "podman") is True
    assert called == []                      # nothing to stop


def test_stop_router_running_success(monkeypatch):
    monkeypatch.setattr(headless, "container_state", lambda name, binary: "running")
    monkeypatch.setattr(headless, "_run",
                        lambda argv: subprocess.CompletedProcess(argv, 0, "", ""))
    assert headless.stop_router(_router(), "podman") is True


def test_stop_router_failure(monkeypatch):
    monkeypatch.setattr(headless, "container_state", lambda name, binary: "running")
    monkeypatch.setattr(headless, "_run",
                        lambda argv: subprocess.CompletedProcess(argv, 1, "", "nope"))
    assert headless.stop_router(_router(), "podman") is False


def test_router_status_ready_dials_loopback(monkeypatch):
    seen = {}
    monkeypatch.setattr(headless, "container_state", lambda name, binary: "running")
    def fake_probe(port, host="127.0.0.1", **kw):
        seen["host"] = host
        return "ready"
    monkeypatch.setattr(headless, "probe_health", fake_probe)
    assert headless.router_status(_router(bind="0.0.0.0"), "podman") == "running"
    assert seen["host"] == "127.0.0.1"       # 0.0.0.0 dialed as loopback


def test_router_status_not_running_is_stopped(monkeypatch):
    monkeypatch.setattr(headless, "container_state", lambda name, binary: "absent")
    # probe must not even be called when the container isn't running
    monkeypatch.setattr(headless, "probe_health",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("probed")))
    assert headless.router_status(_router(), "podman") == "stopped"


def test_wait_ready_returns_true_when_ready(monkeypatch):
    monkeypatch.setattr(headless, "probe_health", lambda port, host="127.0.0.1", **k: "ready")
    assert headless.wait_ready("127.0.0.1", 8080, timeout=5) is True


def test_wait_ready_times_out(monkeypatch):
    monkeypatch.setattr(headless, "probe_health", lambda port, host="127.0.0.1", **k: "loading")
    ticks = iter([0.0, 0.5, 1.0, 1.5])        # monotonic advances past the deadline
    monkeypatch.setattr(headless.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(headless.time, "sleep", lambda s: None)
    assert headless.wait_ready("127.0.0.1", 8080, timeout=1.0, interval=0.5) is False
