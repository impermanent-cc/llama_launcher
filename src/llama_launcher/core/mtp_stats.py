import re
from dataclasses import dataclass


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
    """Return the LAST complete 'draft acceptance = …' line in text, or None."""
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


_BLOCKS = "▁▂▃▄▅▆▇█"


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
