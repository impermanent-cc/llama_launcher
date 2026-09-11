"""Offload sweep decisions: the knob to vary, the counts to run, the memory
lines llama-server logs at load, and the winning point."""

import re
from dataclasses import dataclass

from .vram import positive_int

_MIB = 1024 * 1024
_LINE = re.compile(
    r"^\s*(?:\S+ [A-Z] )?\w+:\s+(?P<dev>\S+)\s+"
    r"(?:(?P<kind>model|KV|RS|compute|output) )?buffer size\s*=\s*"
    r"(?P<mib>[0-9]+(?:\.[0-9]+)?)\s*MiB",
    re.MULTILINE,
)
_CARD = re.compile(r"^CUDA(\d+)$")
_KINDS = {
    None: "model",
    "model": "model",
    "KV": "kv",
    "RS": "kv",
    "compute": "compute",
    "output": "output",
}


@dataclass(frozen=True)
class MeasuredMemory:
    cards: tuple
    ram: dict


@dataclass(frozen=True)
class SweepPoint:
    count: int
    status: str
    ready_seconds: float | None
    rows: tuple
    measured: MeasuredMemory | None
    estimated_cards: tuple
    estimated_ram: int
    error: str


@dataclass(frozen=True)
class Sweep:
    profile: str
    knob: str
    timestamp: str
    points: tuple
    bench_cfg: dict


def sweep_knob(is_moe: bool) -> str:
    """The CPU offload flag a sweep varies: experts on a MoE model, dense
    FFN layers otherwise."""
    return "n-cpu-moe" if is_moe else "n-cpu-ffn"


def sweep_counts(start: int, stop: int, step: int, n_layers) -> list:
    """Counts from start to stop inclusive in steps of step (a step below 1
    means 1), clamped to the layer count, never empty: a stop below start
    yields start alone."""
    start = max(0, int(start))
    stop = max(start, int(stop))
    step = max(1, int(step))
    if n_layers:
        stop = min(stop, int(n_layers))
        start = min(start, int(n_layers))
    return list(range(start, stop + 1, step)) or [start]


def default_range(smallest_fitting) -> tuple:
    """(from, to, step) prefill: from the smallest fitting count, or 0 when
    everything fits or nothing does, eight counts up in steps of two."""
    start = int(smallest_fitting) if smallest_fitting else 0
    return (start, start + 8, 2)


def parse_load_log(text: str) -> MeasuredMemory:
    """Per-device buffer sizes from llama-server's load-time log lines. A
    device named CUDA<n> is card n; every other device counts as RAM. A
    buffer line with no kind word (ik_llama.cpp's model line) counts as
    model; a recurrent-state (RS) buffer line counts into KV. A card has no
    separate output figure, so a card's output buffer line adds into that
    card's compute; an output line on any other device still counts into RAM
    output."""
    cards: dict = {}
    ram = {"model": 0, "kv": 0, "compute": 0, "output": 0}
    for m in _LINE.finditer(text or ""):
        nbytes = int(float(m.group("mib")) * _MIB)
        kind = _KINDS[m.group("kind")]
        card = _CARD.match(m.group("dev"))
        if card:
            i = int(card.group(1))
            slot = cards.setdefault(i, {"model": 0, "kv": 0, "compute": 0})
            target = "compute" if kind == "output" else kind
            if target in slot:
                slot[target] += nbytes
        else:
            ram[kind] += nbytes
    n = max(cards) + 1 if cards else 0
    return MeasuredMemory(
        tuple(cards.get(i, {"model": 0, "kv": 0, "compute": 0}) for i in range(n)), ram
    )


def parse_prompt_sizes(text: str) -> list | None:
    """Comma-separated prompt sizes typed into a benchmark or sweep row: a
    blank string is no sizes at all; any token that is not a positive int
    fails the whole list rather than silently dropping it."""
    text = (text or "").strip()
    if not text:
        return []
    sizes = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        size = positive_int(token)
        if size is None:
            return None
        sizes.append(size)
    return sizes


def last_log_line(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def measured_card_total(m: MeasuredMemory, i: int) -> int:
    if i < 0 or i >= len(m.cards):
        return 0
    c = m.cards[i]
    return c["model"] + c["kv"] + c["compute"]


def measured_ram_total(m: MeasuredMemory) -> int:
    return m.ram["model"] + m.ram["kv"] + m.ram["compute"] + m.ram["output"]


def largest_row(rows):
    """The benchmark row for the largest prompt size, the last of equals;
    None when there are no rows. The one place the largest-row tie is
    decided, for every reader of a point's rows."""
    ordered = sorted(rows or (), key=lambda r: r.get("target_size", 0))
    return ordered[-1] if ordered else None


def _gen_at_largest(point) -> float:
    row = largest_row(point.rows)
    return float(row.get("gen_tok_s", 0.0)) if row is not None else 0.0


def best_point(points):
    """The ok point with the highest generation speed at the largest prompt
    size; the first such point on a tie; None with no ok point."""
    ok = [p for p in points if p.status == "ok" and p.rows]
    return max(ok, key=_gen_at_largest, default=None) if ok else None
