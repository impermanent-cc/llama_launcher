import requests

from llama_launcher.core.prometheus import parse_metrics


def fetch_metrics(port, timeout: float = 1.0) -> dict:
    try:
        r = requests.get(f"http://127.0.0.1:{port}/metrics", timeout=timeout)
        if r.status_code != 200:
            return {}
        return parse_metrics(r.text)
    except requests.RequestException:
        return {}


def fetch_slots(port, timeout: float = 1.0) -> list:
    try:
        r = requests.get(f"http://127.0.0.1:{port}/slots", timeout=timeout)
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
