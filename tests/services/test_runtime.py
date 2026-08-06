import llama_launcher.services.runtime as rt


class FakeCompleted:
    def __init__(self, stdout="", returncode=0):
        self.stdout = stdout
        self.returncode = returncode


def test_binary_available_true(monkeypatch):
    monkeypatch.setattr(rt.shutil, "which", lambda b: "/usr/bin/" + b)
    assert rt.binary_available("podman") is True


def test_binary_available_false(monkeypatch):
    monkeypatch.setattr(rt.shutil, "which", lambda b: None)
    assert rt.binary_available("podman") is False


def test_container_state_running(monkeypatch):
    monkeypatch.setattr(rt, "_run",
                        lambda args: FakeCompleted(stdout="true\n", returncode=0))
    assert rt.container_state("llama-x", "podman") == "running"


def test_container_state_stopped(monkeypatch):
    monkeypatch.setattr(rt, "_run",
                        lambda args: FakeCompleted(stdout="false\n", returncode=0))
    assert rt.container_state("llama-x", "podman") == "stopped"


def test_container_state_absent(monkeypatch):
    monkeypatch.setattr(rt, "_run",
                        lambda args: FakeCompleted(stdout="", returncode=125))
    assert rt.container_state("llama-x", "podman") == "absent"


def test_stop_invokes_binary(monkeypatch):
    captured = {}
    monkeypatch.setattr(rt, "_run", lambda args: captured.setdefault("args", args))
    rt.stop("llama-x", "podman")
    assert captured["args"] == ["podman", "stop", "llama-x"]


def test_stop_argv():
    assert rt.stop_argv("llama-x", "podman") == ["podman", "stop", "-t", "10", "llama-x"]
    assert rt.stop_argv("llama-x", "docker", timeout=3) == ["docker", "stop", "-t", "3", "llama-x"]


def test_is_rootless_true(monkeypatch):
    monkeypatch.setattr(rt, "_run", lambda args: FakeCompleted(stdout="true\n"))
    assert rt.is_rootless("podman") is True


def test_run_oserror_returns_completed_process(monkeypatch):
    monkeypatch.setattr(rt.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no podman")))
    result = rt._run(["podman", "info"])
    assert result.returncode != 0
    assert result.returncode == 127


def test_is_rootless_returns_false_when_binary_missing(monkeypatch):
    monkeypatch.setattr(rt.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no podman")))
    assert rt.is_rootless("podman") is False


def test_started_at_returns_timestamp(monkeypatch):
    monkeypatch.setattr(rt, "_run",
                        lambda args: FakeCompleted(stdout="2024-01-15T10:30:00.123456789Z\n", returncode=0))
    result = rt.started_at("llama-x", "podman")
    assert result == "2024-01-15T10:30:00.123456789Z"


def test_started_at_returns_none_on_failure(monkeypatch):
    monkeypatch.setattr(rt, "_run",
                        lambda args: FakeCompleted(stdout="", returncode=125))
    assert rt.started_at("llama-x", "podman") is None


def test_started_at_returns_none_on_empty_output(monkeypatch):
    monkeypatch.setattr(rt, "_run",
                        lambda args: FakeCompleted(stdout="   \n", returncode=0))
    assert rt.started_at("llama-x", "podman") is None


import json

from llama_launcher.services import runtime

PS_JSON = json.dumps([
    {"Names": ["llama-host"], "State": "running",
     "Labels": {"llama-launcher.profile": "Host", "llama-launcher.mode": "router"}},
    {"Names": ["llama-solo"], "State": "exited",
     "Labels": {"llama-launcher.profile": "Solo", "llama-launcher.mode": "server"}},
])


def test_parse_ps_json_extracts_name_state_and_labels():
    rows = runtime.parse_ps_json(PS_JSON)
    assert rows == [
        {"name": "llama-host", "running": True, "profile": "Host", "mode": "router"},
        {"name": "llama-solo", "running": False, "profile": "Solo", "mode": "server"},
    ]


def test_parse_ps_json_falls_back_to_name_prefix_for_unlabelled():
    # Containers created before labels existed still need to be adoptable.
    out = json.dumps([{"Names": ["llama-old"], "State": "running", "Labels": {}}])
    [row] = runtime.parse_ps_json(out)
    assert row["profile"] == "old"
    assert row["mode"] == "server"


def test_parse_ps_json_skips_foreign_containers():
    out = json.dumps([{"Names": ["postgres"], "State": "running", "Labels": {}}])
    assert runtime.parse_ps_json(out) == []


def test_parse_ps_json_handles_name_string_form():
    out = json.dumps([{"Names": "llama-x", "State": "running", "Labels": {}}])
    assert runtime.parse_ps_json(out)[0]["name"] == "llama-x"


def test_parse_ps_json_tolerates_garbage():
    assert runtime.parse_ps_json("") == []
    assert runtime.parse_ps_json("not json") == []


def test_list_launcher_containers_uses_label_filter(monkeypatch):
    seen = {}

    def fake_run(args):
        seen["args"] = args
        import subprocess
        return subprocess.CompletedProcess(args, 0, PS_JSON, "")

    monkeypatch.setattr(runtime, "_run", fake_run)
    rows = runtime.list_launcher_containers("podman")
    assert "--filter" in seen["args"]
    assert "-a" in seen["args"]
    assert rows[0]["profile"] == "Host"


def test_list_launcher_containers_returns_empty_on_failure(monkeypatch):
    import subprocess
    monkeypatch.setattr(runtime, "_run",
                        lambda args: subprocess.CompletedProcess(args, 1, "", "boom"))
    assert runtime.list_launcher_containers("podman") == []


def test_rm_argv():
    assert runtime.rm_argv("llama-host", "podman") == ["podman", "rm", "-f", "llama-host"]
