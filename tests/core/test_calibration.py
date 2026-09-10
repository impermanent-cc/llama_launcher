"""The estimate against the owner's measured runs.

The fit rule for every calibration record: KV plus recurrent state reads high
by at most a quarter and never low over the sum across cards, or over the RAM
figure for a record with no card figures, and per card within the largest
layer's KV of that band; the per-card compute buffers, the host compute
buffer and the output buffer, each compared on its own so no term can cover
another's shortfall, never read low and read at most two and a half times
high, on every record.
"""

from types import SimpleNamespace

import pytest

from llama_launcher.core import vram
from llama_launcher.core.settings_catalog import CATALOG, accepts
from tests.core.calibration_records import RECORDS, estimate_for

COMPUTE_TOLERANCE = 2.5
KV_TOLERANCE = 1.25
# The load log prints buffer sizes to two decimals of a MiB, so every measured
# figure carries that much rounding on either side of the true byte count.
LOG_PRECISION = 1024 * 1024 // 100


def layer_kv_bytes(record, ctx):
    """The largest single layer's KV cache at the record's head sizes, cache
    types and context, a window layer priced at its window: the slack a
    card's KV figure is allowed, since a layer lands whole on one card."""
    meta = record["meta"]
    ns = SimpleNamespace(**meta)
    n = int(meta["n_layers"])
    heads = meta["n_head_kv"] or meta["n_head"]
    layer_heads = meta.get("kv_layer_heads")
    settings = record["settings"]
    engine = record["engine"]
    k_q = settings.get("cache-type-k", "f16")
    v_q = settings.get("cache-type-v", "f16")
    windowed = vram.window_layer_mask(ns, n)
    cached = vram.kv_layer_mask(ns, n)
    window = int(meta.get("sliding_window") or 0)
    swa_full = accepts(CATALOG["swa-full"], engine) and bool(settings.get("swa-full"))
    if window and any(windowed) and not swa_full and engine != "ik_llama.cpp":
        unified = accepts(CATALOG["kv-unified"], engine) and bool(
            settings.get("kv-unified")
        )
        batch = int(settings.get("batch-size") or CATALOG["batch-size"].default)
        ubatch = int(settings.get("ubatch-size") or CATALOG["ubatch-size"].default)
        batch = min(batch, int(ctx))
        ubatch = min(ubatch, batch)
        w_tokens = vram.window_tokens(
            ctx, window, ubatch, vram.slot_count(settings, engine), unified
        )
    else:
        w_tokens = int(ctx)
    sizes = []
    for il in range(n):
        if not cached[il]:
            continue
        nh = heads
        if layer_heads and il < len(layer_heads) and int(layer_heads[il]):
            nh = int(layer_heads[il])
        if windowed[il]:
            dk = meta.get("head_dim_k_swa") or meta["head_dim_k"]
            dv = meta.get("head_dim_v_swa") or meta["head_dim_v"]
            sizes.append(
                vram.kv_side_bytes(1, nh, dk, w_tokens, k_q)
                + vram.kv_side_bytes(1, nh, dv, w_tokens, v_q)
            )
        else:
            sizes.append(
                vram.kv_side_bytes(1, nh, meta["head_dim_k"], ctx, k_q)
                + vram.kv_side_bytes(1, nh, meta["head_dim_v"], ctx, v_q)
            )
    return max(sizes, default=0)


def test_layer_kv_bytes_returns_the_largest_cached_layer_not_the_smallest():
    """layer_kv_bytes picks the biggest cached layer's KV size: a
    two-layer header with a per-layer head count array returns the size of
    the layer with more heads, not the one with fewer."""
    meta = dict(
        n_layers=2,
        n_head=4,
        n_head_kv=None,
        kv_layer_heads=(2, 8),
        full_attention_interval=None,
        shared_kv_layers=None,
        sliding_window=None,
        sliding_window_pattern=None,
        head_dim_k=64,
        head_dim_v=64,
        head_dim_k_swa=None,
        head_dim_v_swa=None,
    )
    record = {"meta": meta, "settings": {}, "engine": "llama.cpp"}
    ctx = 1000
    larger_layer = vram.kv_side_bytes(1, 8, 64, ctx, "f16") * 2
    smaller_layer = vram.kv_side_bytes(1, 2, 64, ctx, "f16") * 2
    assert larger_layer > smaller_layer
    assert layer_kv_bytes(record, ctx) == larger_layer


def assert_within(estimated, measured, tolerance, what):
    """The estimate reads at least the measurement and at most `tolerance`
    times it, both bounds carrying the log's printed precision."""
    assert measured - LOG_PRECISION <= estimated, f"{what} {estimated} under {measured}"
    assert estimated <= tolerance * measured + LOG_PRECISION, (
        f"{what} {estimated} over {tolerance} x {measured}"
    )


@pytest.mark.parametrize("rec", RECORDS, ids=[r["name"] for r in RECORDS])
def test_records_carry_no_output_card_key(rec):
    assert "output_card" not in rec


PENDING_FIGURES = {"compute", "host", "output"}


@pytest.mark.parametrize("rec", RECORDS, ids=[r["name"] for r in RECORDS])
def test_record_pending_names_only_the_figures_the_bands_know(rec):
    """A record's `pending` entry names only figures the compute test
    checks: a typo or a figure no test compares would otherwise skip
    nothing and pass silently."""
    assert set(rec.get("pending", ())) <= PENDING_FIGURES


@pytest.mark.parametrize("rec", RECORDS, ids=[r["name"] for r in RECORDS])
def test_record_kv_reads_high_by_at_most_a_quarter_and_never_low(rec):
    est = estimate_for(rec)
    cards = rec["measured"]["cards"]
    if not cards:
        measured = rec["measured"]["ram"]["kv"]
        estimated = est.ram.kv + est.ram.state
        assert_within(estimated, measured, KV_TOLERANCE, f"{rec['name']} ram kv")
        return
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
    if "compute" in rec.get("pending", ()):
        pytest.skip(f"{rec['name']}: compute figure pending a refit")
    est = estimate_for(rec)
    cards = rec["measured"]["cards"]
    measured = sorted(m["compute"] for m in cards)
    estimated = sorted(c.compute for c in est.cards[: len(cards)])
    for i, (m, c) in enumerate(zip(measured, estimated, strict=True)):
        assert_within(c, m, COMPUTE_TOLERANCE, f"{rec['name']} sorted compute {i}")


@pytest.mark.parametrize("rec", RECORDS, ids=[r["name"] for r in RECORDS])
def test_record_host_buffer_never_reads_low_and_at_most_two_and_a_half_high(rec):
    if "host" in rec.get("pending", ()):
        pytest.skip(f"{rec['name']}: host buffer figure pending a refit")
    est = estimate_for(rec)
    r = rec["measured"]["ram"]
    assert_within(
        est.ram.host, r["compute"], COMPUTE_TOLERANCE, f"{rec['name']} host buffer"
    )


@pytest.mark.parametrize("rec", RECORDS, ids=[r["name"] for r in RECORDS])
def test_record_output_buffer_never_reads_low_and_at_most_two_and_a_half_high(rec):
    if "output" in rec.get("pending", ()):
        pytest.skip(f"{rec['name']}: output buffer figure pending a refit")
    est = estimate_for(rec)
    r = rec["measured"]["ram"]
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
