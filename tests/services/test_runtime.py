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


def test_is_rootless_true(monkeypatch):
    monkeypatch.setattr(rt, "_run", lambda args: FakeCompleted(stdout="true\n"))
    assert rt.is_rootless("podman") is True
