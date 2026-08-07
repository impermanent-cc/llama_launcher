from dataclasses import dataclass, field


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
    gen = sum(_tok_s(s.get("predicted_n", 0), s.get("predicted_ms", 0)) for s in samples) / n
    total = sum((s.get("prompt_ms", 0) + s.get("predicted_ms", 0)) / 1000.0 for s in samples) / n
    return BenchmarkRow(target_size, samples[-1].get("prompt_n", 0), pp, gen, total)
