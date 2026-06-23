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
    monkeypatch.setattr(met.requests, "get", lambda url, timeout=None: R())
    m = met.fetch_metrics(8080)
    assert m["llamacpp:predicted_tokens_seconds"] == 9.0


def test_fetch_metrics_error_returns_empty(monkeypatch):
    def boom(url, timeout=None):
        raise met.requests.RequestException("nope")
    monkeypatch.setattr(met.requests, "get", boom)
    assert met.fetch_metrics(8080) == {}
