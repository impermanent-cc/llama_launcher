import json

import pytest

from llama_launcher.services import router_api


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", lines=()):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text
        self._lines = lines

    def json(self):
        return self._payload

    def iter_lines(self, decode_unicode=False):
        for line in self._lines:
            yield line

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_base_url():
    assert router_api.base_url("127.0.0.1", 8080) == "http://127.0.0.1:8080"


def test_auth_headers_omitted_without_key():
    assert router_api.auth_headers(None) == {}
    assert router_api.auth_headers("") == {}


def test_auth_headers_use_bearer():
    assert router_api.auth_headers("sk-x") == {"Authorization": "Bearer sk-x"}


def test_list_models_never_autoloads(monkeypatch):
    seen = {}

    def fake_get(url, headers=None, timeout=None, params=None, **kw):
        seen["url"] = url
        seen["params"] = params
        return FakeResponse(payload={"data": [{"id": "a", "status": {"value": "sleeping"}}]})

    monkeypatch.setattr(router_api.requests, "get", fake_get)
    models = router_api.list_models("127.0.0.1", 8080, "sk-x")
    # Observing a model must never load it — that would defeat idle unloading.
    assert seen["params"]["autoload"] == "false"
    assert models[0].status == "sleeping"


def test_list_models_returns_none_when_unreachable(monkeypatch):
    # None (can't reach it) must be distinguishable from [] (up, serving
    # nothing) -- it is the only status signal an unattended host has.
    def boom(*a, **kw):
        raise router_api.requests.RequestException("down")

    monkeypatch.setattr(router_api.requests, "get", boom)
    assert router_api.list_models("127.0.0.1", 8080, None) is None


def test_list_models_returns_none_on_non_200(monkeypatch):
    monkeypatch.setattr(router_api.requests, "get",
                        lambda *a, **kw: FakeResponse(status_code=401))
    assert router_api.list_models("127.0.0.1", 8080, None) is None


def test_list_models_returns_empty_list_for_a_reachable_empty_router(monkeypatch):
    monkeypatch.setattr(router_api.requests, "get",
                        lambda *a, **kw: FakeResponse(payload={"data": []}))
    assert router_api.list_models("127.0.0.1", 8080, None) == []


def test_load_model_posts_model_name(monkeypatch):
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None, **kw):
        seen["url"] = url
        seen["json"] = json
        return FakeResponse(payload={"success": True})

    monkeypatch.setattr(router_api.requests, "post", fake_post)
    assert router_api.load_model("127.0.0.1", 8080, "sk-x", "qwen") is True
    assert seen["url"].endswith("/models/load")
    assert seen["json"] == {"model": "qwen"}


def test_unload_model_posts_to_unload(monkeypatch):
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None, **kw):
        seen["url"] = url
        return FakeResponse(payload={"success": True})

    monkeypatch.setattr(router_api.requests, "post", fake_post)
    assert router_api.unload_model("127.0.0.1", 8080, "sk-x", "qwen") is True
    assert seen["url"].endswith("/models/unload")


def test_load_model_false_on_error(monkeypatch):
    def boom(*a, **kw):
        raise router_api.requests.RequestException("down")

    monkeypatch.setattr(router_api.requests, "post", boom)
    assert router_api.load_model("127.0.0.1", 8080, None, "q") is False


def test_iter_sse_events_yields_parsed_events(monkeypatch):
    frame = json.dumps({"model": "q", "event": "model_status",
                        "data": {"status": "loaded"}})
    lines = [f"data: {frame}", "", ": ping", "", f"data: {frame}", ""]
    monkeypatch.setattr(router_api.requests, "get",
                        lambda *a, **kw: FakeResponse(lines=lines))

    events = list(router_api.iter_sse_events("127.0.0.1", 8080, "sk-x"))
    assert [e.event for e in events] == ["model_status", "model_status"]
    assert events[0].data == {"status": "loaded"}
