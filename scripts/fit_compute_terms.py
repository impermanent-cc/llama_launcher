"""Grid search over COMPUTE_TERMS and the ik engine scale against the
calibration records.

One search over the six terms and the ik_llama.cpp scale together: the point
with the smallest worst-case over-estimate that reads low on no record's
figure. Fitting the terms on the mainline records first and the scale
afterwards misses points this one finds, since a mainline optimum need not
leave the ik record a scale that reads high.

Every figure is compared on its own so no term can cover another's shortfall:
each card's compute buffer (sorted, since the output card follows settings
the records do not carry), the host compute buffer and the output buffer. The
KV ratios printed at the end are fixed by the layer mask and placement, so no
point of either grid moves them.

The feasible region is narrow against the box a single coarse grid can search
in sane time, so the search runs in stages: a coarse pass over a wide box
locates the region, then two refinement passes narrow around the previous
stage's best point with finer steps. Every compared figure must read at
least the measurement and at most two and a half times it, and each
refinement box is built from the step sizes that tolerance calls for (about
1.0 on residual and ssm, 0.02 then 0.01 on logits and vocab, 0.04 then 0.02
on host, 0.01 on the ik scale) rather than hand-picked bounds, so the
stages stay reproducible from the coarse pass alone. FFN carries forward as
a single value once the coarse pass settles it at its physical floor, since
raising it only makes every figure read higher.

Run from the repository root: .venv/bin/python scripts/fit_compute_terms.py
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_launcher.core import vram
from tests.core.calibration_records import RECORDS, estimate_for

COARSE_GRID = {
    "logits": [0.0, 0.25, 0.5, 0.75, 1.0],
    # One f32 copy of the FFN intermediate per token is the physical floor.
    "ffn": [1.0, 2.0, 4.0],
    "residual": [2.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0],
    "ssm": [0.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0],
    "vocab": [0.0, 0.2, 0.35, 0.5, 0.75, 1.0],
    "host": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
}
IK_ENGINE = "ik_llama.cpp"
COARSE_IK_SCALES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.78, 0.82, 0.9]
# Step size and half-width (number of steps either side of the previous
# stage's best point) for each refinement stage.
REFINEMENT_STAGES = (
    {
        "logits": (0.02, 3),
        "residual": (1.0, 6),
        "ssm": (1.0, 4),
        "vocab": (0.02, 3),
        "host": (0.04, 3),
    },
    {
        "logits": (0.01, 3),
        "residual": (1.0, 3),
        "ssm": (1.0, 3),
        "vocab": (0.01, 3),
        "host": (0.02, 3),
    },
)
REFINEMENT_SCALE_STEP = 0.01
REFINEMENT_SCALE_HALF_WIDTH = 6
# The tolerance on the compute and buffer figures: never low, at most two
# and a half times high.
TOLERANCE = 2.5
# The load log prints buffer sizes to two decimals of a MiB, the rounding
# every measured figure carries.
LOG_PRECISION = 1024 * 1024 // 100

MAINLINE = [r for r in RECORDS if r["engine"] != IK_ENGINE]
IK = [r for r in RECORDS if r["engine"] == IK_ENGINE]


def fitted_figures(record):
    """(name, estimated, measured) for every figure the coefficients move:
    the sorted per-card compute buffers, the host compute buffer and the
    output buffer, leaving out any figure the record marks pending, since no
    coefficient the grid searches can move a figure a record marks that way."""
    pending = record.get("pending", ())
    est = estimate_for(record)
    out = []
    if "compute" not in pending:
        cards = record["measured"]["cards"]
        measured = sorted(m["compute"] for m in cards)
        estimated = sorted(c.compute for c in est.cards[: len(cards)])
        out.extend(
            (f"{record['name']} sorted compute {i}", c, m)
            for i, (m, c) in enumerate(zip(measured, estimated, strict=True))
        )
    r = record["measured"]["ram"]
    if "host" not in pending:
        out.append((f"{record['name']} host buffer", est.ram.host, r["compute"]))
    if "output" not in pending:
        out.append((f"{record['name']} output buffer", est.ram.output, r["output"]))
    return out


def kv_ratios(record):
    """Estimate over measured for the KV plus recurrent state summed across
    cards and per card, which no coefficient in the grid changes. A record
    whose measured KV is zero contributes no ratio for that figure, since
    nothing is divisible by it."""
    est = estimate_for(record)
    cards = record["measured"]["cards"]
    if not cards:
        measured = record["measured"]["ram"]["kv"]
        if measured == 0:
            return []
        return [(f"{record['name']} ram kv", (est.ram.kv + est.ram.state) / measured)]
    out = [
        (
            f"{record['name']} card {i} kv",
            (est.cards[i].kv + est.cards[i].state) / m["kv"],
        )
        for i, m in enumerate(cards)
        if m["kv"] != 0
    ]
    measured = sum(m["kv"] for m in cards)
    if measured != 0:
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


def search(grid, ik_scales):
    """(feasible, best_effort) over the terms and the ik scale together. Each
    is (score, limiting figure, terms, scale, figures) or None: the feasible
    one scores its worst ratio, the best-effort one its largest deviation
    from the measurement in either direction. main reads best_effort only
    when feasible is None, so the deviation and its update are computed only
    while no feasible point has been found yet."""
    feasible = None
    best_effort = None
    for combo in itertools.product(*grid.values()):
        terms = dict(zip(grid.keys(), combo, strict=True))
        vram.COMPUTE_TERMS.update(terms)
        mainline = figures_over(MAINLINE)
        for scale in ik_scales:
            vram.ENGINE_COMPUTE_SCALE[IK_ENGINE] = scale
            figures = mainline + figures_over(IK)
            ratio, name = worst_of(figures)
            if reads_high(figures) and (feasible is None or ratio < feasible[0]):
                feasible = (ratio, name, terms, scale, figures)
            if feasible is None:
                low = min(est / meas for _, est, meas in figures)
                span = max(ratio, 1.0 / low)
                if best_effort is None or span < best_effort[0]:
                    best_effort = (span, name, terms, scale, figures)
    return feasible, best_effort


def around(center, step, half_width, floor=0.0):
    """`2 * half_width + 1` values spaced `step` apart around `center`, none
    below `floor`."""
    values = {
        round(max(floor, center + i * step), 6)
        for i in range(-half_width, half_width + 1)
    }
    return sorted(values)


def refine(point, scale, stage):
    """The grid and ik scales for one refinement stage, boxed around the
    previous stage's best point at that stage's step sizes. `ffn` carries
    forward as the single value the previous stage settled on."""
    grid = {
        name: around(point[name], step, half_width)
        for name, (step, half_width) in stage.items()
    }
    grid["ffn"] = [point["ffn"]]
    ik_scales = around(
        scale, REFINEMENT_SCALE_STEP, REFINEMENT_SCALE_HALF_WIDTH, floor=0.0
    )
    return grid, ik_scales


def print_figures(figures):
    for name, est, meas in figures:
        print(f"  {est / meas:8.3f}  {name}")


def main():
    original_terms = dict(vram.COMPUTE_TERMS)
    original_scale = dict(vram.ENGINE_COMPUTE_SCALE)

    print(f"coarse pass: {len(COARSE_GRID)} terms, {len(COARSE_IK_SCALES)} ik scales")
    feasible, best_effort = search(COARSE_GRID, COARSE_IK_SCALES)
    for stage in REFINEMENT_STAGES:
        best = feasible if feasible is not None else best_effort
        point, scale = best[2], best[3]
        grid, ik_scales = refine(point, scale, stage)
        print(f"refining around {point} ik scale {scale}")
        stage_feasible, stage_best_effort = search(grid, ik_scales)
        if stage_feasible is not None and (
            feasible is None or stage_feasible[0] < feasible[0]
        ):
            feasible = stage_feasible
        if feasible is None and stage_best_effort is not None:
            best_effort = stage_best_effort

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
    print(f"tolerance {TOLERANCE}: {'met' if worst <= TOLERANCE else 'NOT met'}")
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
