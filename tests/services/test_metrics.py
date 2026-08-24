import llama_launcher.services.metrics as met
from llama_launcher.services.metrics import kv_usage_ratio, kv_ratio


def test_kv_usage_ratio_takes_busiest_slot_not_summed():
    # KV% is a per-slot occupancy, so it reports the fullest slot -- NOT the old
    # behaviour of summing tokens over one denominator of all slots' ctx summed.
    slots = [{"n_ctx": 100, "n_prompt_tokens": 20},
             {"n_ctx": 100, "n_prompt_tokens": 30}]
    assert abs(kv_usage_ratio(slots) - 0.30) < 1e-9


def test_kv_usage_ratio_active_slot_not_diluted_by_idle_ctx():
    # The old code summed every slot's n_ctx into the denominator, so three idle
    # slots dragged a half-full active slot's KV% down toward 0. It must instead
    # report the active (is_processing) slot's own occupancy.
    slots = [
        {"n_ctx": 4096, "is_processing": False},
        {"n_ctx": 4096, "is_processing": False},
        {"n_ctx": 4096, "is_processing": False},
        {"n_ctx": 4096, "is_processing": True, "n_prompt_tokens": 2048},
    ]
    assert abs(kv_usage_ratio(slots) - 0.50) < 1e-9


def test_kv_usage_ratio_uses_full_sequence_length():
    # Occupancy is the whole resident sequence (n_prompt_tokens), not just the
    # tokens newly processed this turn (n_prompt_tokens_processed reads ~0 once
    # prefill is done and the slot is generating).
    slots = [{"n_ctx": 100, "is_processing": True,
              "n_prompt_tokens": 80, "n_prompt_tokens_processed": 3}]
    assert abs(kv_usage_ratio(slots) - 0.80) < 1e-9


def test_kv_usage_ratio_empty():
    assert kv_usage_ratio([]) is None


def test_decode_rate_computes_live_tok_s():
    # Δ(n_decode_total)/Δt is the live generation rate: 23 tokens over 1.0s = 23 tok/s.
    assert abs(met.decode_rate((100, 10.0), (123, 11.0)) - 23.0) < 1e-9


def test_decode_rate_none_without_prior_read():
    assert met.decode_rate(None, (5, 1.0)) is None


def test_decode_rate_none_when_counter_did_not_move():
    # Idle server: the counter is static, so there is no rate to report (not 0).
    assert met.decode_rate((50, 1.0), (50, 2.0)) is None


def test_decode_rate_none_on_counter_reset():
    # Server restarted -> counter went backwards; don't report a negative rate.
    assert met.decode_rate((900, 1.0), (5, 2.0)) is None


def test_kv_ratio_prefers_metric_over_slots():
    # llama.cpp's own kv_cache_usage_ratio gauge reflects true cache occupancy;
    # it must win over the slots-derived prompt-token estimate.
    slots = [{"n_ctx": 100, "n_prompt_tokens_processed": 25}]   # would be 0.25
    assert kv_ratio({"llamacpp:kv_cache_usage_ratio": 0.7}, slots) == 0.7


def test_kv_ratio_falls_back_to_slots_when_metric_absent():
    slots = [{"n_ctx": 100, "n_prompt_tokens_processed": 25}]
    assert abs(kv_ratio({}, slots) - 0.25) < 1e-9


def test_kv_ratio_none_when_both_absent():
    assert kv_ratio({}, []) is None


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


def test_counter_rate_is_the_generic_delta_rate():
    """decode_rate's counter-delta math, exposed generically so the live
    prompt rate (prefill progress counter) shares one implementation."""
    assert abs(met.counter_rate((0, 0.0), (10, 2.0)) - 5.0) < 1e-9
    assert met.counter_rate(None, (10, 2.0)) is None
    assert met.counter_rate((10, 1.0), (10, 2.0)) is None


def test_prompt_progress_takes_processing_slot():
    slots = [{"is_processing": True, "n_prompt_tokens_processed": 1200},
             {"is_processing": False, "n_prompt_tokens_processed": 50}]
    assert met.prompt_progress(slots) == 1200


def test_prompt_progress_max_across_processing_slots():
    slots = [{"is_processing": True, "n_prompt_tokens_processed": 300},
             {"is_processing": True, "n_prompt_tokens_processed": 900}]
    assert met.prompt_progress(slots) == 900


def test_prompt_progress_none_when_nothing_processing():
    assert met.prompt_progress([]) is None
    assert met.prompt_progress([{"is_processing": False,
                                 "n_prompt_tokens_processed": 50}]) is None
    assert met.prompt_progress([{"is_processing": True}]) is None


def test_fetch_metrics_brackets_ipv6_host(monkeypatch):
    seen = {}
    class R:
        status_code = 200
        text = ""
    def _get(url, **kw):
        seen["url"] = url
        return R()
    monkeypatch.setattr(met.requests, "get", _get)
    met.fetch_metrics_text(8080, host="::1")
    assert seen["url"] == "http://[::1]:8080/metrics"
