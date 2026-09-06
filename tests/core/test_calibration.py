"""The estimate against the owner's measured runs.

The fit rule for every calibration record: KV plus recurrent state reads high
by at most a quarter and never low over the sum across cards, and per card
within one layer's KV of that band; the per-card compute buffers, the host
compute buffer and the output buffer, each compared on its own so no term can
cover another's shortfall, never read low and read at most two and a half
times high.
"""

from types import SimpleNamespace

import pytest

from llama_launcher.core import vram
from tests.core.calibration_records import RECORDS

COMPUTE_TOLERANCE = 2.5
KV_TOLERANCE = 1.25
# The load log prints buffer sizes to two decimals of a MiB, so every measured
# figure carries that much rounding on either side of the true byte count.
LOG_PRECISION = 1024 * 1024 // 100


def estimate_for(record):
    """The record's estimate, with one byte of weights standing for the model
    so placement runs on the record's settings alone."""
    meta = SimpleNamespace(**record["meta"])
    return vram.estimate_memory(
        meta,
        1,
        settings=record["settings"],
        engine=record["engine"],
        free_bytes_per_gpu=record["free_bytes_per_gpu"],
    )


def layer_kv_bytes(record, ctx):
    """One layer's KV cache at the record's head sizes, cache types and
    context: the slack a card's KV figure is allowed, since a layer lands
    whole on one card or the other."""
    meta = record["meta"]
    heads = meta["n_head_kv"] or meta["n_head"]
    settings = record["settings"]
    return vram.kv_cache_bytes(
        1,
        heads,
        meta["head_dim_k"],
        ctx,
        settings.get("cache-type-k", "f16"),
        settings.get("cache-type-v", "f16"),
    )


def assert_within(estimated, measured, tolerance, what):
    """The estimate reads at least the measurement and at most `tolerance`
    times it, both bounds carrying the log's printed precision."""
    assert measured - LOG_PRECISION <= estimated, f"{what} {estimated} under {measured}"
    assert estimated <= tolerance * measured + LOG_PRECISION, (
        f"{what} {estimated} over {tolerance} x {measured}"
    )


@pytest.mark.parametrize("rec", RECORDS, ids=[r["name"] for r in RECORDS])
def test_record_kv_reads_high_by_at_most_a_quarter_and_never_low(rec):
    est = estimate_for(rec)
    cards = rec["measured"]["cards"]
    if not cards:
        pytest.skip(
            "no card figures: this record's KV sits in RAM as the labelled "
            "sliding-window upper bound, which the measurement does not match"
        )
    measured = sum(m["kv"] for m in cards)
    estimated = sum(c.kv + c.state for c in est.cards[: len(cards)])
    assert_within(estimated, measured, KV_TOLERANCE, f"{rec['name']} kv sum")
    slack = layer_kv_bytes(rec, est.ctx) + LOG_PRECISION
    for i, m in enumerate(cards):
        kv = est.cards[i].kv + est.cards[i].state
        assert m["kv"] - slack <= kv <= KV_TOLERANCE * m["kv"] + slack, (
            f"{rec['name']} card {i} kv {kv} against {m['kv']} within {slack}"
        )


@pytest.mark.parametrize("rec", RECORDS, ids=[r["name"] for r in RECORDS])
def test_record_compute_never_reads_low_and_at_most_two_and_a_half_high(rec):
    est = estimate_for(rec)
    cards = rec["measured"]["cards"]
    measured = sorted(m["compute"] for m in cards)
    estimated = sorted(c.compute for c in est.cards[: len(cards)])
    for i, (m, c) in enumerate(zip(measured, estimated, strict=True)):
        assert_within(c, m, COMPUTE_TOLERANCE, f"{rec['name']} sorted compute {i}")
    r = rec["measured"]["ram"]
    assert_within(
        est.ram.host, r["compute"], COMPUTE_TOLERANCE, f"{rec['name']} host buffer"
    )
    assert_within(
        est.ram.output, r["output"], COMPUTE_TOLERANCE, f"{rec['name']} output buffer"
    )


def test_ik_record_cards_fit_the_free_vram_the_run_had():
    """The ik record loaded on its two cards, so the estimate of everything
    but the weights, plus the model bytes the log measured on each card,
    stays inside that card's free VRAM. The context checkpoints the server
    keeps as host vectors are charged to RAM, which is what leaves the cards
    room."""
    rec = next(r for r in RECORDS if r["engine"] == "ik_llama.cpp")
    est = estimate_for(rec)
    assert est.ram.checkpoints > 0
    for i, measured in enumerate(rec["measured"]["cards"]):
        total = est.cards[i].total + measured["model"]
        free = rec["free_bytes_per_gpu"][i]
        assert total <= free, f"{rec['name']} card {i}: {total} over free {free}"
