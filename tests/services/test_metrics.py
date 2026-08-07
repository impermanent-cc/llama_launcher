import llama_launcher.services.metrics as met
from llama_launcher.services.metrics import kv_usage_ratio


def test_kv_usage_ratio():
    slots = [{"n_ctx": 100, "n_prompt_tokens_processed": 20},
             {"n_ctx": 100, "n_prompt_tokens_processed": 30}]
    assert abs(kv_usage_ratio(slots) - 0.25) < 1e-9


def test_kv_usage_ratio_empty():
    assert kv_usage_ratio([]) is None


def test_fetch_metrics_parses(monkeypatch):
    class R:
        status_code = 200
        text = "llamacpp:predicted_tokens_seconds 9.0\n"
        def raise_for_status(self): pass
    monkeypatch.setattr(met.requests, "get", lambda url, timeout=None, **kw: R())
    m = met.fetch_metrics(8080)
    assert m["llamacpp:predicted_tokens_seconds"] == 9.0


def test_fetch_metrics_error_returns_empty(monkeypatch):
    def boom(url, timeout=None, **kw):
        raise met.requests.RequestException("nope")
    monkeypatch.setattr(met.requests, "get", boom)
    assert met.fetch_metrics(8080) == {}


from llama_launcher.services import metrics


class _Resp:
    def __init__(self, status_code=200, text="", payload=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload or []

    def json(self):
        return self._payload


def test_fetch_metrics_scopes_to_model_without_autoloading(monkeypatch):
    seen = {}

    def fake_get(url, timeout=None, params=None, headers=None, **kw):
        seen["url"] = url
        seen["params"] = params
        seen["headers"] = headers
        return _Resp(text="llamacpp:n_decode_total 3\n")

    monkeypatch.setattr(metrics.requests, "get", fake_get)
    out = metrics.fetch_metrics(8080, model="qwen", api_key="sk-x")
    assert seen["params"] == {"model": "qwen", "autoload": "false"}
    assert seen["headers"] == {"Authorization": "Bearer sk-x"}
    assert out["llamacpp:n_decode_total"] == 3.0


def test_fetch_metrics_without_model_sends_no_params(monkeypatch):
    seen = {}

    def fake_get(url, timeout=None, params=None, headers=None, **kw):
        seen["params"] = params
        return _Resp(text="")

    monkeypatch.setattr(metrics.requests, "get", fake_get)
    metrics.fetch_metrics(8080)
    assert seen["params"] is None


def test_fetch_metrics_text_returns_raw_body(monkeypatch):
    monkeypatch.setattr(metrics.requests, "get",
                        lambda *a, **kw: _Resp(text="raw body"))
    assert metrics.fetch_metrics_text(8080) == "raw body"


def test_fetch_metrics_uses_supplied_host(monkeypatch):
    seen = {}

    def fake_get(url, **kw):
        seen["url"] = url
        return _Resp(text="")

    monkeypatch.setattr(metrics.requests, "get", fake_get)
    metrics.fetch_metrics(8080, host="192.168.1.9")
    assert seen["url"].startswith("http://192.168.1.9:8080")


import requests
from llama_launcher.core.props import PropsInfo


class _RespProps:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload
    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def test_fetch_props_returns_parsed_on_200(monkeypatch):
    monkeypatch.setattr(metrics.requests, "get",
                        lambda *a, **k: _RespProps(200, {"build_info": "b9755", "total_slots": 1}))
    info = metrics.fetch_props(8080)
    assert isinstance(info, PropsInfo)
    assert info.build == "b9755" and info.total_slots == 1


def test_fetch_props_none_on_503(monkeypatch):
    monkeypatch.setattr(metrics.requests, "get", lambda *a, **k: _RespProps(503))
    assert metrics.fetch_props(8080) is None


def test_fetch_props_none_on_bad_json(monkeypatch):
    monkeypatch.setattr(metrics.requests, "get", lambda *a, **k: _RespProps(200, None))
    assert metrics.fetch_props(8080) is None


def test_fetch_props_none_on_connection_error(monkeypatch):
    def boom(*a, **k):
        raise requests.RequestException("refused")
    monkeypatch.setattr(metrics.requests, "get", boom)
    assert metrics.fetch_props(8080) is None
