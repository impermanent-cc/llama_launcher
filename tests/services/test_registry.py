from llama_launcher.services.registry import (
    latest_build_tag, split_image, variant_prefix,
)


def test_latest_build_tag_picks_highest():
    tags = ["server-cuda12-b9600", "server-cuda12-b9755", "server-cuda12-b9628",
            "server-cuda13-b9999", "server-cuda12", "buildcache-amd64-b9999"]
    assert latest_build_tag(tags, "server-cuda12") == "server-cuda12-b9755"


def test_latest_build_tag_exact_prefix_only():
    # "server-cuda" must NOT match "server-cuda12-*"
    tags = ["server-cuda-b10", "server-cuda12-b9999"]
    assert latest_build_tag(tags, "server-cuda") == "server-cuda-b10"


def test_latest_build_tag_none_when_no_match():
    assert latest_build_tag(["server-vulkan-b1"], "server-rocm") is None


def test_split_image():
    assert split_image("ghcr.io/ggml-org/llama.cpp:server-cuda12-b9628") == (
        "ghcr.io/ggml-org/llama.cpp", "server-cuda12-b9628")


def test_variant_prefix():
    assert variant_prefix("server-cuda12-b9628") == "server-cuda12"
    assert variant_prefix("server") == "server"


def test_fetch_latest_paginates(monkeypatch):
    import llama_launcher.services.registry as reg

    class FakeResp:
        def __init__(self, payload, link="", token=None):
            self._payload = payload
            self.headers = {"Link": link} if link else {}
            self._token = token
        def json(self):
            return {"token": self._token} if self._token else self._payload
        def raise_for_status(self):
            pass

    calls = {"n": 0}

    def fake_get(url, headers=None, timeout=None):
        if "token" in url:
            return FakeResp({}, token="t")
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResp({"tags": ["server-cuda12-b1"]},
                            link='</v2/ggml-org/llama.cpp/tags/list?last=x&n=1000>; rel="next"')
        return FakeResp({"tags": ["server-cuda12-b2"]})

    monkeypatch.setattr(reg.requests, "get", fake_get)
    assert reg.fetch_latest("ghcr.io/ggml-org/llama.cpp", "server-cuda12") == "server-cuda12-b2"
