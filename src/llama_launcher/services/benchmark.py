import os
from dataclasses import dataclass, field

from llama_launcher.services import metrics


@dataclass
class BenchmarkRow:
    target_size: int
    prompt_n: int
    pp_tok_s: float
    gen_tok_s: float
    total_s: float


@dataclass
class BenchmarkRun:
    timestamp: str
    sizes: list
    n_predict: int
    warmup: int
    repeats: int
    rows: list = field(default_factory=list)
    snapshot: dict = field(default_factory=dict)


_FILLER_WORD = "lorem "


def filler_prompt(target_tokens: int) -> str:
    """Deterministic text of ~target_tokens tokens (~0.75 words/token)."""
    words = max(1, int(target_tokens * 0.75))
    return (_FILLER_WORD * words).strip()


def _tok_s(n, ms) -> float:
    return (n / (ms / 1000.0)) if ms and ms > 0 else 0.0


def row_from_timings(target_size: int, samples: list) -> BenchmarkRow:
    n = len(samples)
    pp = sum(_tok_s(s.get("prompt_n", 0), s.get("prompt_ms", 0)) for s in samples) / n
    gen = (
        sum(_tok_s(s.get("predicted_n", 0), s.get("predicted_ms", 0)) for s in samples)
        / n
    )
    total = (
        sum(
            (s.get("prompt_ms", 0) + s.get("predicted_ms", 0)) / 1000.0 for s in samples
        )
        / n
    )
    return BenchmarkRow(target_size, samples[-1].get("prompt_n", 0), pp, gen, total)


class BenchmarkError(Exception):
    pass


def run_benchmark(
    client, sizes, n_predict, warmup, repeats, snapshot, timestamp, should_cancel=None
):
    rows = []
    for size in sizes:
        prompt = filler_prompt(size)
        samples = []
        for i in range(warmup + repeats):
            if should_cancel is not None and should_cancel():
                raise BenchmarkError("cancelled")
            try:
                resp = client(prompt, n_predict)
            except Exception as e:  # any transport/parse error aborts
                raise BenchmarkError(str(e)) from e
            if i >= warmup:  # discard the warmup hits
                samples.append(resp.get("timings", {}))
        rows.append(row_from_timings(size, samples))
    return BenchmarkRun(
        timestamp, list(sizes), n_predict, warmup, repeats, rows, snapshot
    )


_SNAP_FLAGS = {
    "ngl": "n-gpu-layers",
    "fa": "flash-attn",
    "ctk": "cache-type-k",
    "ctv": "cache-type-v",
    "ctx": "ctx-size",
    "ncmoe": "n-cpu-moe",
    "sm": "split-mode",
}


def build_snapshot(profile, member=None) -> dict:
    # A router whose loaded model resolves to no member records blanks: the
    # router profile's own settings/model are leftover single-server form
    # fields (a router Save keeps every set widget), values the members never
    # ran with -- honest Nones beat a confidently wrong config row.
    if member is None and getattr(profile, "mode", "server") == "router":
        src, model = None, ""
    else:
        src = member if member is not None else profile
        model = getattr(member, "model", None) or getattr(profile, "model", None) or ""
    snap = {
        "model": os.path.basename(model) if model else None,
        "image": profile.image,
        "mode": profile.mode,
    }
    for short, key in _SNAP_FLAGS.items():
        v = src.settings.get(key) if src is not None else None
        snap[short] = None if v is None else str(v)
    return snap


def requests_client(host, port, api_key, model_scope, timeout=300.0):
    import requests

    def _call(prompt, n_predict):
        r = requests.post(
            f"http://{host}:{port}/completion",
            json={
                "prompt": prompt,
                "temperature": 0,
                "n_predict": n_predict,
                "stream": False,
            },
            headers=metrics._headers(api_key),
            params=metrics._scope(model_scope),
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()

    return _call
