from llama_launcher.services import lora_api


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else []

    def json(self):
        return self._payload


def test_base_url_and_auth_headers():
    assert lora_api.base_url("127.0.0.1", 8080) == "http://127.0.0.1:8080"
    assert lora_api.auth_headers(None) == {}
    assert lora_api.auth_headers("") == {}
    assert lora_api.auth_headers("sk-x") == {"Authorization": "Bearer sk-x"}


def test_list_adapters_parses_the_payload(monkeypatch):
    monkeypatch.setattr(lora_api.requests, "get", lambda *a, **kw: FakeResponse(
        payload=[{"id": 0, "path": "a.gguf", "scale": 0.5}]))
    got = lora_api.list_adapters("127.0.0.1", 8080, "sk-x")
    assert [(a.id, a.scale) for a in got] == [(0, 0.5)]


def test_list_adapters_sends_the_key(monkeypatch):
    seen = {}

    def fake_get(url, headers=None, timeout=None, **kw):
        seen["url"] = url
        seen["headers"] = headers
        return FakeResponse(payload=[])

    monkeypatch.setattr(lora_api.requests, "get", fake_get)
    lora_api.list_adapters("h", 9, "sk-x")
    assert seen["url"] == "http://h:9/lora-adapters"
    assert seen["headers"]["Authorization"] == "Bearer sk-x"


def test_list_adapters_none_when_unreachable_but_empty_when_up(monkeypatch):
    # [] (up, launched without adapters) must stay distinguishable from None
    # (no answer): the panel says something different for each.
    def boom(*a, **kw):
        raise lora_api.requests.RequestException("down")

    monkeypatch.setattr(lora_api.requests, "get", boom)
    assert lora_api.list_adapters("127.0.0.1", 8080, None) is None

    monkeypatch.setattr(lora_api.requests, "get",
                        lambda *a, **kw: FakeResponse(payload=[]))
    assert lora_api.list_adapters("127.0.0.1", 8080, None) == []


def test_list_adapters_none_on_non_200(monkeypatch):
    monkeypatch.setattr(lora_api.requests, "get",
                        lambda *a, **kw: FakeResponse(status_code=401))
    assert lora_api.list_adapters("127.0.0.1", 8080, None) is None


def test_set_scales_posts_every_id_with_an_explicit_scale(monkeypatch):
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None, **kw):
        seen["url"] = url
        seen["json"] = json
        return FakeResponse()

    monkeypatch.setattr(lora_api.requests, "post", fake_post)
    assert lora_api.set_scales("h", 9, None, {1: 0.8, 0: 0.0}) is True
    assert seen["url"] == "http://h:9/lora-adapters"
    # Sorted by id, and 0.0 is sent rather than omitted: upstream treats an
    # absent adapter ambiguously, so the request states every scale outright.
    assert seen["json"] == [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.8}]


def test_set_scales_coerces_types(monkeypatch):
    seen = {}
    monkeypatch.setattr(lora_api.requests, "post",
                        lambda url, headers=None, json=None, timeout=None, **kw:
                        (seen.update(json=json), FakeResponse())[1])
    lora_api.set_scales("h", 9, None, {0: 1})
    assert seen["json"] == [{"id": 0, "scale": 1.0}]
    assert isinstance(seen["json"][0]["scale"], float)


def test_set_scales_false_on_failure(monkeypatch):
    monkeypatch.setattr(lora_api.requests, "post",
                        lambda *a, **kw: FakeResponse(status_code=500))
    assert lora_api.set_scales("h", 9, None, {0: 1.0}) is False

    def boom(*a, **kw):
        raise lora_api.requests.RequestException("down")

    monkeypatch.setattr(lora_api.requests, "post", boom)
    assert lora_api.set_scales("h", 9, None, {0: 1.0}) is False
