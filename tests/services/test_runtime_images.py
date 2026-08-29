import subprocess
from llama_launcher.services import runtime


def test_parse_images_detailed():
    out = ("llama-custom:x-20260828|2.1 GB|2026-08-28 10:00:00 +0000 UTC\n"
           "<none>:<none>|1 GB|whenever\n"
           "garbage line\n")
    d = runtime.parse_images_detailed(out)
    assert list(d) == ["llama-custom:x-20260828"]
    assert d["llama-custom:x-20260828"].size == "2.1 GB"


def test_remove_image_success_and_failure(monkeypatch):
    calls = []
    def fake_run(args, timeout=30):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
    monkeypatch.setattr(runtime, "_run", fake_run)
    ok, err = runtime.remove_image("podman", "t:1")
    assert ok and err == ""
    assert calls[0][-2:] == ["rmi", "t:1"]
    monkeypatch.setattr(runtime, "_run", lambda a, timeout=30: subprocess.CompletedProcess(a, 125, "", "in use"))
    ok, err = runtime.remove_image("podman", "t:1")
    assert not ok and "in use" in err
