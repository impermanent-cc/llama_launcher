import requests

from llama_launcher.core.prometheus import parse_metrics
from llama_launcher.core.props import PropsInfo, parse_props


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


def url_host(host: str) -> str:
    """Bracket a bare IPv6 literal for use in a URL authority. `::1` -> `[::1]`;
    an already-bracketed host or a name/IPv4 is returned unchanged. Without this,
    an IPv6 bind_host (an address the app itself accepts) makes a malformed URL."""
    if host and ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def fetch_metrics_text(port, timeout: float = 1.0, model: str | None = None,
                       api_key: str | None = None, host: str = "127.0.0.1") -> str:
    """Raw /metrics body (needed for label-aware parsing), '' on any failure."""
    try:
        r = requests.get(f"http://{url_host(host)}:{port}/metrics", timeout=timeout,
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
        r = requests.get(f"http://{url_host(host)}:{port}/slots", timeout=timeout,
                         params=_scope(model), headers=_headers(api_key))
        if r.status_code != 200:
            return []
        data = r.json()
        return data if isinstance(data, list) else []
    except (requests.RequestException, ValueError):
        return []


def fetch_props(port, timeout: float = 1.0, api_key: str | None = None,
                host: str = "127.0.0.1") -> PropsInfo | None:
    """Parsed /props, or None on any failure.

    /props is server-wide (not model-scoped) and is exempt from auth and the
    idle timer; the key is passed for symmetry and does no harm.
    """
    try:
        r = requests.get(f"http://{url_host(host)}:{port}/props", timeout=timeout,
                         headers=_headers(api_key))
        if r.status_code != 200:
            return None
        return parse_props(r.json())
    except (requests.RequestException, ValueError):
        return None


def _slot_occupancy_ratio(s: dict) -> float | None:
    """One slot's KV occupancy: resident sequence length / that slot's own ctx.

    Occupancy is the whole resident sequence (`n_prompt_tokens`), not just the
    tokens newly processed this turn (`n_prompt_tokens_processed`, which reads
    ~0 once prefill finishes and the slot is generating). Older schemas expose
    only the latter, so it's the fallback. None when the slot has no ctx.
    """
    n_ctx = s.get("n_ctx", 0)
    if n_ctx <= 0:
        return None
    used = s.get("n_prompt_tokens")
    if used is None:
        used = s.get("n_prompt_tokens_processed", 0)
    return used / n_ctx


def kv_usage_ratio(slots: list) -> float | None:
    """Slots-derived KV occupancy: the busiest slot's fill, never a sum.

    KV is per-slot, so the figure is the fullest slot's occupancy against its
    OWN context -- not tokens summed over a denominator of every slot's ctx
    added together, which lets idle slots drag a busy slot's KV% down toward 0.
    Prefers a processing slot (what the user is
    watching) over an idle one holding a stale sequence. None if no slot has ctx.
    """
    active = [r for s in slots if s.get("is_processing")
              for r in (_slot_occupancy_ratio(s),) if r is not None]
    if active:
        return max(active)
    resident = [r for s in slots
                for r in (_slot_occupancy_ratio(s),) if r is not None]
    return max(resident) if resident else None


def counter_rate(prev: tuple | None, cur: tuple | None) -> float | None:
    """Live rate from two (counter, monotonic_time) reads. None when there's
    no prior read, no forward movement (idle), or the counter went backwards
    (server restarted / a per-request counter reset between requests)."""
    if prev is None or cur is None:
        return None
    (prev_count, prev_t), (cur_count, cur_t) = prev, cur
    d_count, d_t = cur_count - prev_count, cur_t - prev_t
    if d_count <= 0 or d_t <= 0:
        return None
    return d_count / d_t


def decode_rate(prev: tuple | None, cur: tuple | None) -> float | None:
    """Live generation tok/s from two (n_decode_total, monotonic_time) reads.

    llama.cpp's `predicted_tokens_seconds` gauge (and `tokens_predicted_total`)
    only update when a request COMPLETES, so they read 0 throughout an in-flight
    generation. `n_decode_total` instead increments ~1 per generated token in
    real time, so the delta between two reads is a live throughput.
    """
    return counter_rate(prev, cur)


def prompt_progress(slots: list) -> int | None:
    """Prefill progress counter: the busiest PROCESSING slot's
    `n_prompt_tokens_processed`. It grows batch by batch through a prefill
    (and reads ~0 once generation starts), so the delta between two reads is
    a live prompt tok/s -- the `prompt_tokens_seconds` gauge only updates at
    request completion. None when nothing is processing (its per-request reset
    would otherwise register as a bogus negative delta).
    """
    vals = [s.get("n_prompt_tokens_processed") for s in slots
            if s.get("is_processing")]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def kv_ratio(m: dict, slots: list) -> float | None:
    """KV-cache occupancy, preferring llama.cpp's own gauge.

    `llamacpp:kv_cache_usage_ratio` from /metrics reflects real cache occupancy
    and holds steady while a context is resident. The slots-derived estimate
    (n_prompt_tokens_processed / n_ctx) only tracks momentary prompt processing
    and reads 0 whenever the server isn't mid-prefill, so it's the fallback used
    only when the metric is absent (e.g. --metrics off).
    """
    metric = m.get("llamacpp:kv_cache_usage_ratio")
    if metric is not None:
        return metric
    return kv_usage_ratio(slots)
