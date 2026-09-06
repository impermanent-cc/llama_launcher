"""Grid search over COMPUTE_TERMS and the ik engine scale against the
calibration records.

One search over the five terms and the ik_llama.cpp scale together: the point
with the smallest worst-case over-estimate that reads low on no record's
figure. Fitting the terms on the mainline records first and the scale
afterwards misses points this one finds, since a mainline optimum need not
leave the ik record a scale that reads high.

Every figure is compared on its own so no term can cover another's shortfall:
each card's compute buffer (sorted, since the output card follows settings
the records do not carry), the host compute buffer and the output buffer. The
KV ratios printed at the end are fixed by the layer mask and placement, so no
point of either grid moves them.

Run from the repository root: .venv/bin/python scripts/fit_compute_terms.py
"""

import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_launcher.core import vram
from tests.core.calibration_records import RECORDS

GRID = {
    "logits": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    # One f32 copy of the FFN intermediate per token is the physical floor.
    "ffn": [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
    "residual": [2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0],
    "ssm": [8.0, 16.0, 24.0, 32.0, 48.0, 64.0, 80.0, 96.0],
    "host": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
}
IK_ENGINE = "ik_llama.cpp"
IK_SCALES = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
# The tolerance on the compute and buffer figures: never low, at most two
# and a half times high.
TOLERANCE = 2.5
# The load log prints buffer sizes to two decimals of a MiB, the rounding
# every measured figure carries.
LOG_PRECISION = 1024 * 1024 // 100

MAINLINE = [r for r in RECORDS if r["engine"] != IK_ENGINE]
IK = [r for r in RECORDS if r["engine"] == IK_ENGINE]


def estimate_for(record):
    meta = SimpleNamespace(**record["meta"])
    return vram.estimate_memory(
        meta,
        1,
        settings=record["settings"],
        engine=record["engine"],
        free_bytes_per_gpu=record["free_bytes_per_gpu"],
    )


def fitted_figures(record):
    """(name, estimated, measured) for every figure the coefficients move:
    the sorted per-card compute buffers, the host compute buffer and the
    output buffer."""
    est = estimate_for(record)
    cards = record["measured"]["cards"]
    measured = sorted(m["compute"] for m in cards)
    estimated = sorted(c.compute for c in est.cards[: len(cards)])
    out = [
        (f"{record['name']} sorted compute {i}", c, m)
        for i, (m, c) in enumerate(zip(measured, estimated, strict=True))
    ]
    r = record["measured"]["ram"]
    out.append((f"{record['name']} host buffer", est.ram.host, r["compute"]))
    out.append((f"{record['name']} output buffer", est.ram.output, r["output"]))
    return out


def kv_ratios(record):
    """Estimate over measured for the KV plus recurrent state summed across
    cards and per card, which no coefficient in the grid changes."""
    est = estimate_for(record)
    cards = record["measured"]["cards"]
    if not cards:
        return []
    out = [
        (
            f"{record['name']} card {i} kv",
            (est.cards[i].kv + est.cards[i].state) / m["kv"],
        )
        for i, m in enumerate(cards)
    ]
    measured = sum(m["kv"] for m in cards)
    estimated = sum(c.kv + c.state for c in est.cards[: len(cards)])
    out.append((f"{record['name']} kv sum", estimated / measured))
    return out


def figures_over(records):
    return [triple for rec in records for triple in fitted_figures(rec)]


def reads_high(figures):
    """Whether every figure reaches its measurement, within the log's
    printed precision."""
    return all(est >= meas - LOG_PRECISION for _, est, meas in figures)


def worst_of(figures):
    """(ratio, name) of the figure that reads highest."""
    return max(((est / meas, name) for name, est, meas in figures))


def search():
    """(feasible, best_effort) over the terms and the ik scale together. Each
    is (score, limiting figure, terms, scale, figures) or None: the feasible
    one scores its worst ratio, the best-effort one its largest deviation
    from the measurement in either direction."""
    feasible = None
    best_effort = None
    for combo in itertools.product(*GRID.values()):
        terms = dict(zip(GRID.keys(), combo, strict=True))
        vram.COMPUTE_TERMS.update(terms)
        mainline = figures_over(MAINLINE)
        for scale in IK_SCALES:
            vram.ENGINE_COMPUTE_SCALE[IK_ENGINE] = scale
            figures = mainline + figures_over(IK)
            ratio, name = worst_of(figures)
            if reads_high(figures) and (feasible is None or ratio < feasible[0]):
                feasible = (ratio, name, terms, scale, figures)
            low = min(est / meas for _, est, meas in figures)
            span = max(ratio, 1.0 / low)
            if best_effort is None or span < best_effort[0]:
                best_effort = (span, name, terms, scale, figures)
    return feasible, best_effort


def print_figures(figures):
    for name, est, meas in figures:
        print(f"  {est / meas:8.3f}  {name}")


def main():
    original_terms = dict(vram.COMPUTE_TERMS)
    original_scale = dict(vram.ENGINE_COMPUTE_SCALE)
    feasible, best_effort = search()
    if feasible is None:
        print("no grid point reads high on every figure")
        print(f"least-bad point: {best_effort[2]} ik scale {best_effort[3]}")
        print_figures(best_effort[4])
        for name, est, meas in best_effort[4]:
            if est < meas - LOG_PRECISION:
                print(f"  READS LOW {est / meas:.3f}  {name}")
        vram.COMPUTE_TERMS.update(original_terms)
        vram.ENGINE_COMPUTE_SCALE.update(original_scale)
        return
    worst, limiting, terms, scale, figures = feasible
    print(f"terms: {terms}")
    print(f"ik scale: {scale}")
    print_figures(figures)
    print(f"WORST RATIO: {worst:.3f}")
    print(f"LIMITING FIGURE: {limiting}")
    print(
        f"SPEC 2.19 tolerance {TOLERANCE}: {'met' if worst <= TOLERANCE else 'NOT met'}"
    )
    vram.COMPUTE_TERMS.update(terms)
    vram.ENGINE_COMPUTE_SCALE[IK_ENGINE] = scale
    print("kv ratios (fixed by the mask and placement, not by these terms):")
    for rec in RECORDS:
        for name, ratio in kv_ratios(rec):
            print(f"  {ratio:8.3f}  {name}")
    vram.COMPUTE_TERMS.update(original_terms)
    vram.ENGINE_COMPUTE_SCALE.update(original_scale)


if __name__ == "__main__":
    main()
