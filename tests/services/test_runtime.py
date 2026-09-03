import json

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
    monkeypatch.setattr(
        rt, "_run", lambda args: FakeCompleted(stdout="true\n", returncode=0)
    )
    assert rt.container_state("llama-x", "podman") == "running"


def test_container_state_stopped(monkeypatch):
    monkeypatch.setattr(
        rt, "_run", lambda args: FakeCompleted(stdout="false\n", returncode=0)
    )
    assert rt.container_state("llama-x", "podman") == "stopped"


def test_container_state_absent(monkeypatch):
    monkeypatch.setattr(
        rt, "_run", lambda args: FakeCompleted(stdout="", returncode=125)
    )
    assert rt.container_state("llama-x", "podman") == "absent"


def test_stop_invokes_binary(monkeypatch):
    captured = {}
    monkeypatch.setattr(rt, "_run", lambda args: captured.setdefault("args", args))
    rt.stop("llama-x", "podman")
    assert captured["args"] == ["podman", "stop", "llama-x"]


def test_stop_argv():
    assert rt.stop_argv("llama-x", "podman") == [
        "podman",
        "stop",
        "-t",
        "10",
        "llama-x",
    ]
    assert rt.stop_argv("llama-x", "docker", timeout=3) == [
        "docker",
        "stop",
        "-t",
        "3",
        "llama-x",
    ]


def test_is_rootless_true(monkeypatch):
    monkeypatch.setattr(rt, "_run", lambda args: FakeCompleted(stdout="true\n"))
    assert rt.is_rootless("podman") is True


def test_is_rootless_docker_reads_security_options(monkeypatch):
    # docker has no Host.Security.Rootless field (that podman template errors and
    # leaves stdout empty, which would read as always False). A rootless
    # docker daemon lists "name=rootless" among SecurityOptions instead.
    calls = {}

    def fake_run(args):
        calls["args"] = args
        return FakeCompleted(
            stdout="name=apparmor\nname=seccomp,profile=default\nname=rootless\n"
        )

    monkeypatch.setattr(rt, "_run", fake_run)
    assert rt.is_rootless("docker") is True
    assert calls["args"][0] == "docker" and "info" in calls["args"]


def test_is_rootless_docker_rootful(monkeypatch):
    monkeypatch.setattr(
        rt,
        "_run",
        lambda args: FakeCompleted(
            stdout="name=apparmor\nname=seccomp,profile=default\n"
        ),
    )
    assert rt.is_rootless("docker") is False


def test_run_oserror_returns_completed_process(monkeypatch):
    monkeypatch.setattr(
        rt.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no podman")),
    )
    result = rt._run(["podman", "info"])
    assert result.returncode != 0
    assert result.returncode == 127


def test_run_timeout_returns_completed_process_not_hang(monkeypatch):
    """A hung podman must never block indefinitely: _run bounds the call with a
    timeout and, when it fires, returns a failing CompletedProcess (returncode
    != 0) instead of raising or hanging. update_status runs these on the UI
    thread, so an unbounded call freezes the GUI forever."""
    import subprocess as sp

    def _boom(*a, **k):
        assert k.get("timeout") is not None  # a timeout must be passed
        raise sp.TimeoutExpired(cmd=a[0] if a else "podman", timeout=k["timeout"])

    monkeypatch.setattr(rt.subprocess, "run", _boom)
    result = rt._run(["podman", "stats"])
    assert result.returncode != 0
    assert result.stdout == ""


def test_run_passes_timeout_to_subprocess(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        rt.subprocess,
        "run",
        lambda *a, **k: (
            captured.update(k) or rt.subprocess.CompletedProcess(a, 0, "", "")
        ),
    )
    rt._run(["podman", "ps"])
    assert captured.get("timeout") is not None


def test_is_rootless_returns_false_when_binary_missing(monkeypatch):
    monkeypatch.setattr(
        rt.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no podman")),
    )
    assert rt.is_rootless("podman") is False


def test_started_at_returns_timestamp(monkeypatch):
    monkeypatch.setattr(
        rt,
        "_run",
        lambda args: FakeCompleted(
            stdout="2024-01-15T10:30:00.123456789Z\n", returncode=0
        ),
    )
    result = rt.started_at("llama-x", "podman")
    assert result == "2024-01-15T10:30:00.123456789Z"


def test_started_at_returns_none_on_failure(monkeypatch):
    monkeypatch.setattr(
        rt, "_run", lambda args: FakeCompleted(stdout="", returncode=125)
    )
    assert rt.started_at("llama-x", "podman") is None


def test_started_at_returns_none_on_empty_output(monkeypatch):
    monkeypatch.setattr(
        rt, "_run", lambda args: FakeCompleted(stdout="   \n", returncode=0)
    )
    assert rt.started_at("llama-x", "podman") is None


PS_JSON = json.dumps(
    [
        {
            "Names": ["llama-host"],
            "State": "running",
            "Labels": {
                "llama-launcher.profile": "Host",
                "llama-launcher.mode": "router",
            },
        },
        {
            "Names": ["llama-solo"],
            "State": "exited",
            "Labels": {
                "llama-launcher.profile": "Solo",
                "llama-launcher.mode": "server",
            },
        },
    ]
)


def test_parse_ps_json_extracts_name_state_and_labels():
    rows = rt.parse_ps_json(PS_JSON)
    assert rows == [
        {"name": "llama-host", "running": True, "profile": "Host", "mode": "router"},
        {"name": "llama-solo", "running": False, "profile": "Solo", "mode": "server"},
    ]


def test_parse_ps_json_falls_back_to_name_prefix_for_unlabelled():
    # Unlabelled containers must still be adoptable.
    out = json.dumps([{"Names": ["llama-old"], "State": "running", "Labels": {}}])
    [row] = rt.parse_ps_json(out)
    assert row["profile"] == "old"
    assert row["mode"] == "server"


def test_parse_ps_json_skips_foreign_containers():
    out = json.dumps([{"Names": ["postgres"], "State": "running", "Labels": {}}])
    assert rt.parse_ps_json(out) == []


def test_parse_ps_json_handles_name_string_form():
    out = json.dumps([{"Names": "llama-x", "State": "running", "Labels": {}}])
    assert rt.parse_ps_json(out)[0]["name"] == "llama-x"


def test_parse_ps_json_tolerates_garbage():
    assert rt.parse_ps_json("") == []
    assert rt.parse_ps_json("not json") == []


def test_list_launcher_containers_uses_label_filter(monkeypatch):
    seen = {}

    def fake_run(args):
        seen["args"] = args
        import subprocess

        return subprocess.CompletedProcess(args, 0, PS_JSON, "")

    monkeypatch.setattr(rt, "_run", fake_run)
    rows = rt.list_launcher_containers("podman")
    assert "--filter" in seen["args"]
    assert "-a" in seen["args"]
    assert rows[0]["profile"] == "Host"


def test_list_launcher_containers_returns_empty_on_failure(monkeypatch):
    import subprocess

    monkeypatch.setattr(
        rt, "_run", lambda args: subprocess.CompletedProcess(args, 1, "", "boom")
    )
    assert rt.list_launcher_containers("podman") == []


def test_rm_argv():
    assert rt.rm_argv("llama-host", "podman") == [
        "podman",
        "rm",
        "-f",
        "llama-host",
    ]
