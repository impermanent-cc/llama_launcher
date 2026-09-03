import re
from dataclasses import dataclass

from .prometheus import parse_labeled_metric, parse_metrics


@dataclass(frozen=True)
class DraftStats:
    acceptance: float
    mean_len: float
    per_position: tuple
    accepted: int
    generated: int


_DRAFT_RE = re.compile(
    r"draft acceptance\s*=\s*([0-9.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated\s*\)"
    r".*?mean acceptance length\s*=\s*([0-9.]+)"
    r".*?acceptance rate per position\s*=\s*\(([^)]*)\)"
)


def parse_draft_stats(text: str) -> DraftStats | None:
    """Return the LAST complete 'draft acceptance = \u2026' line in text, or None."""
    last = None
    for m in _DRAFT_RE.finditer(text):
        last = m
    if last is None:
        return None
    pos = tuple(float(x) for x in last.group(5).split(",") if x.strip())
    return DraftStats(
        acceptance=float(last.group(1)),
        mean_len=float(last.group(4)),
        per_position=pos,
        accepted=int(last.group(2)),
        generated=int(last.group(3)),
    )


_BLOCKS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


def sparkline(values, width: int = 0) -> str:
    """Render numbers as a unicode block sparkline. Empty -> ''. Flat -> all low."""
    vals = list(values)
    if width > 0:
        vals = vals[-width:]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 0:
        return _BLOCKS[0] * len(vals)
    return "".join(_BLOCKS[int((v - lo) / span * (len(_BLOCKS) - 1))] for v in vals)


@dataclass(frozen=True)
class SpecCounters:
    """A point-in-time read of llama.cpp's spec-decode counters.

    These are monotonic counters, so a rate needs two reads; see spec_delta.
    Unlike the log-scraped acceptance line, they can be fetched per-model in
    router mode via /metrics?model=<id>.
    """
    draft_tokens: float
    accepted: float
    drafts: float
    per_position: tuple = ()


_SPEC_PREFIX = "llamacpp:spec_decode_num_"


def spec_counters(text: str) -> SpecCounters | None:
    """Read spec-decode counters from raw /metrics text, or None if absent.

    They are absent whenever speculative decoding is off, and the per-position
    family is absent until the first speculative request completes.
    """
    flat = parse_metrics(text)
    draft_tokens = flat.get(_SPEC_PREFIX + "draft_tokens_total")
    accepted = flat.get(_SPEC_PREFIX + "accepted_tokens_total")
    drafts = flat.get(_SPEC_PREFIX + "drafts_total")
    if draft_tokens is None or accepted is None or drafts is None:
        return None

    per_pos = parse_labeled_metric(
        text, _SPEC_PREFIX + "accepted_tokens_per_pos_total", "position")
    ordered = tuple(
        per_pos[k] for k in sorted(per_pos, key=lambda s: int(s) if s.isdigit() else 0)
    )
    return SpecCounters(draft_tokens=draft_tokens, accepted=accepted,
                        drafts=drafts, per_position=ordered)


def spec_delta(prev: SpecCounters, cur: SpecCounters) -> DraftStats | None:
    """Acceptance between two counter reads, or None if there is nothing to report.

    Returns None when no drafting happened since the last poll, and when the
    counters went backwards (the server restarted).
    """
    d_draft = cur.draft_tokens - prev.draft_tokens
    d_accepted = cur.accepted - prev.accepted
    d_drafts = cur.drafts - prev.drafts
    if d_draft <= 0 or d_drafts <= 0 or d_accepted < 0:
        return None
    return DraftStats(
        acceptance=d_accepted / d_draft,
        mean_len=d_accepted / d_drafts,
        per_position=cur.per_position,
        accepted=int(d_accepted),
        generated=int(d_draft),
    )
