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
