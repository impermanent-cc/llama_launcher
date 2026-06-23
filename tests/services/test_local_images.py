import llama_launcher.services.runtime as rt


class Fake:
    def __init__(self, stdout="", rc=0):
        self.stdout, self.returncode = stdout, rc


def test_parse_images_filters_and_dedupes():
    out = ("ghcr.io/ggml-org/llama.cpp:full\n"
           "ghcr.io/ggml-org/llama.cpp:server\n"
           "<none>:<none>\n"
           "docker.io/library/redis:7\n"
           "ghcr.io/ggml-org/llama.cpp:full\n")   # duplicate
    assert rt.parse_images(out) == [
        "ghcr.io/ggml-org/llama.cpp:full",
        "ghcr.io/ggml-org/llama.cpp:server",
    ]


def test_parse_images_empty():
    assert rt.parse_images("") == []


def test_list_local_images(monkeypatch):
    monkeypatch.setattr(rt, "_run",
                        lambda a: Fake(stdout="ghcr.io/ggml-org/llama.cpp:full\n", rc=0))
    assert rt.list_local_images("podman") == ["ghcr.io/ggml-org/llama.cpp:full"]


def test_list_local_images_uses_images_format(monkeypatch):
    captured = {}

    def fake(a):
        captured["args"] = a
        return Fake(stdout="", rc=0)

    monkeypatch.setattr(rt, "_run", fake)
    rt.list_local_images("docker")
    assert captured["args"][:2] == ["docker", "images"]
    assert "--format" in captured["args"]


def test_list_local_images_error_returns_empty(monkeypatch):
    monkeypatch.setattr(rt, "_run", lambda a: Fake(stdout="", rc=127))
    assert rt.list_local_images("podman") == []
