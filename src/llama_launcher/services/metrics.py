import requests

from llama_launcher.core.prometheus import parse_metrics


_NO_AUTOLOAD = {"autoload": "false"}


def _scope(model: str | None) -> dict | None:
    """Query params for a router-scoped GET, or None in single-model mode.

    `autoload=false` matters: without it, polling /metrics?model=X would LOAD X.
    """
    if not model:
        return None
    return {"model": model, **_NO_AUTOLOAD}


def _headers(api_key: str | None) -> dict | None:
    return {"Authorization": f"Bearer {api_key}"} if api_key else None


def fetch_metrics_text(port, timeout: float = 1.0, model: str | None = None,
                       api_key: str | None = None, host: str = "127.0.0.1") -> str:
    """Raw /metrics body (needed for label-aware parsing), '' on any failure."""
    try:
        r = requests.get(f"http://{host}:{port}/metrics", timeout=timeout,
                         params=_scope(model), headers=_headers(api_key))
        if r.status_code != 200:
            return ""
        return r.text
    except requests.RequestException:
        return ""


def fetch_metrics(port, timeout: float = 1.0, model: str | None = None,
                  api_key: str | None = None, host: str = "127.0.0.1") -> dict:
    text = fetch_metrics_text(port, timeout, model, api_key, host)
    return parse_metrics(text) if text else {}


def fetch_slots(port, timeout: float = 1.0, model: str | None = None,
                api_key: str | None = None, host: str = "127.0.0.1") -> list:
    try:
        r = requests.get(f"http://{host}:{port}/slots", timeout=timeout,
                         params=_scope(model), headers=_headers(api_key))
        if r.status_code != 200:
            return []
        data = r.json()
        return data if isinstance(data, list) else []
    except (requests.RequestException, ValueError):
        return []


def kv_usage_ratio(slots: list) -> float | None:
    total_ctx = sum(s.get("n_ctx", 0) for s in slots)
    if total_ctx <= 0:
        return None
    used = sum(s.get("n_prompt_tokens_processed", 0) for s in slots)
    return used / total_ctx
