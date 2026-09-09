from llama_launcher.core import balance
from llama_launcher.core import placement as pl
from llama_launcher.core.gguf import GgufMeta, TensorInfo
from llama_launcher.core.vram import (
    CardEstimate,
    MemoryEstimate,
    RamEstimate,
    estimate_memory,
)

MIB = 1024 * 1024
GIB = 1024 * MIB


def _tabled_estimate(margins, costs, *, base_ctx, free, default_margin, default_cost):
    """A fake estimate_memory that reads a card's margin and marginal cost
    per token from a table keyed by --tensor-split counts, growing the KV
    figure linearly with context above base_ctx by exactly that cost so
    marginal_bytes_per_token recovers it. Counts absent from the table score
    far worse than any tabled entry, closing off any state beyond the ones
    a test names. The table holds canned scores, not a physical model: a
    caller may assign a state whatever margin and cost it likes, including a
    margin that moves for a card whose own layer count did not change
    between two tabled states, to pin the search's move-selection rule in
    isolation from whether the real estimator could ever produce that
    landscape. A tabled or default margin and cost may each be given as a
    single number for every card alike, rather than a per-card tuple."""

    def fake(meta, weights_bytes, *, settings, engine, **_kwargs):
        raw = settings.get("tensor-split", "")
        key = tuple(int(x) for x in raw.split(",")) if raw else None
        margin = margins.get(key, default_margin)
        cost = costs.get(key, default_cost)
        if not isinstance(margin, (tuple, list)):
            margin = (margin,) * len(free)
        if not isinstance(cost, (tuple, list)):
            cost = (cost,) * len(free)
        ctx = int(settings.get("ctx-size", base_ctx) or base_ctx)
        cards = tuple(
            CardEstimate(
                weights=0,
                kv=(free[i] - margin[i]) + max(0, ctx - base_ctx) * cost[i],
                compute=0,
                overhead=0,
                state=0,
            )
            for i in range(len(free))
        )
        return MemoryEstimate(
            cards=cards,
            ram=RamEstimate(weights=0, kv=0, host=0),
            ctx=ctx,
            kv_upper_bound=False,
        )

    return fake


def _meta(n_layers=8, swa=None, n_head_kv=8, sliding_window_pattern=None):
    ts = [TensorInfo("token_embd.weight", 1, 0, 10 * MIB)]
    for i in range(n_layers):
        ts.append(TensorInfo(f"blk.{i}.attn_q.weight", 1, 0, 100 * MIB))
        ts.append(TensorInfo(f"blk.{i}.ffn_up.weight", 1, 0, 400 * MIB))
    ts.append(TensorInfo("output.weight", 1, 0, 100 * MIB))
    return GgufMeta(
        arch="llama",
        n_layers=n_layers,
        n_head=8,
        n_head_kv=n_head_kv,
        n_embd=512,
        ctx_train=32768,
        n_ff=256,
        n_vocab=1000,
        sliding_window=swa,
        sliding_window_pattern=sliding_window_pattern,
        tensors=tuple(ts),
    )


def _kwargs(free=(16 * GIB, 12 * GIB), **settings):
    return dict(
        settings={"ctx-size": 8192, **settings},
        engine="llama.cpp",
        free_bytes_per_gpu=list(free),
    )


def test_marginal_cost_is_positive_and_per_card():
    """Each card of a multi-GPU layout gets its own bytes-per-token figure:
    the 8 f16 layers split 6 to card 0 and 2 to card 1 at 2048 bytes per
    token each, 12288.0 and 4096.0."""
    cost = balance.marginal_bytes_per_token(_meta(), 0, **_kwargs())
    assert cost == (12288.0, 4096.0)


def test_marginal_cost_is_the_same_rate_with_flash_attention_off():
    """With flash attention off, the attention-scores compute term grows
    with the context while the cache-only rate does not, so the rate stays
    12288.0 and 4096.0."""
    cost = balance.marginal_bytes_per_token(
        _meta(), 0, **_kwargs(**{"flash-attn": "off"})
    )
    assert cost == (12288.0, 4096.0)


def test_marginal_cost_under_a_sliding_window_matches_the_window_rule():
    """A window layer's cache is pinned by the window cap and adds nothing
    per token, so an alternating window pattern halves the rate relative to
    full attention, and --swa-full restores it: no second formula is needed
    for the cap."""
    meta = _meta(swa=512, sliding_window_pattern=tuple(i % 2 == 0 for i in range(8)))
    windowed = balance.marginal_bytes_per_token(meta, 0, **_kwargs())
    assert windowed == (6144.0, 2048.0)
    full = balance.marginal_bytes_per_token(meta, 0, **_kwargs(**{"swa-full": True}))
    assert full == (12288.0, 4096.0)


def test_marginal_cost_uses_the_raw_arg_context_not_the_profile_one():
    """--ctx-size on the raw command line overrides the profile's own
    ctx-size before the estimate is priced, the same way the server reads
    argv. With the 4096-token window against the 8192 to 16384 step, the
    cap binds partway through the step, so the window layers grow at half
    rate over it and the measured figure is three quarters of the full
    rate."""
    meta = _meta(swa=4096, sliding_window_pattern=tuple(i % 2 == 0 for i in range(8)))
    at_profile = balance.marginal_bytes_per_token(meta, 0, **_kwargs())
    assert at_profile == (12288.0, 4096.0)
    at_raw = balance.marginal_bytes_per_token(
        meta, 0, **_kwargs(), raw_args="--ctx-size 16384"
    )
    assert at_raw == (9216.0, 3072.0)


def test_marginal_cost_resolves_ctx_size_absent_or_zero_to_ctx_train():
    """A profile with no --ctx-size, or --ctx-size explicitly 0, prices the
    rate at the model's trained context of 32768 rather than at zero."""
    kw = dict(engine="llama.cpp", free_bytes_per_gpu=list((16 * GIB, 12 * GIB)))
    absent = balance.marginal_bytes_per_token(_meta(), 0, settings={}, **kw)
    assert absent == (12288.0, 4096.0)
    zero = balance.marginal_bytes_per_token(_meta(), 0, settings={"ctx-size": 0}, **kw)
    assert zero == (12288.0, 4096.0)


def test_marginal_cost_scales_with_the_layers_a_card_holds():
    """A card holding more of the model's layers under an explicit
    --tensor-split carries a larger marginal cost per token."""
    kw = _kwargs()
    even = balance.marginal_bytes_per_token(
        _meta(), 0, **{**kw, "settings": {**kw["settings"], "tensor-split": "5,4"}}
    )
    lopsided = balance.marginal_bytes_per_token(
        _meta(), 0, **{**kw, "settings": {**kw["settings"], "tensor-split": "8,1"}}
    )
    assert lopsided[0] > even[0]
    assert lopsided[1] < even[1]


def test_marginal_cost_is_zero_without_a_gpu_cache():
    """A profile that keeps the KV cache off the GPU carries no marginal
    cost per token on any card."""
    cost = balance.marginal_bytes_per_token(
        _meta(), 0, **_kwargs(**{"no-kv-offload": True})
    )
    assert cost == (0.0, 0.0)


def test_marginal_cost_none_when_the_estimate_is_unknowable():
    """A header too thin to support an estimate yields None rather than a
    cost tuple."""
    assert (
        balance.marginal_bytes_per_token(
            GgufMeta(arch="llama", n_layers=0, n_embd=0), 0, **_kwargs()
        )
        is None
    )


def test_gpu_entries_counts_blocks_and_the_output_layer():
    """--n-gpu-layers all puts every block layer and the output layer on a
    card, index n_layers being the output layer; a smaller count keeps
    only the last layers, counted from the output layer down. A raw-arg
    spelling of --n-gpu-layers overrides the profile setting."""
    meta = _meta(n_layers=8)
    entries = balance.gpu_entries(
        meta, settings={"n-gpu-layers": "all"}, engine="llama.cpp"
    )
    assert entries == tuple(range(9))
    part = balance.gpu_entries(meta, settings={"n-gpu-layers": 3}, engine="llama.cpp")
    assert part == (6, 7, 8)
    raw = balance.gpu_entries(
        meta,
        settings={"n-gpu-layers": "all"},
        engine="llama.cpp",
        raw_args="--n-gpu-layers 3",
    )
    assert raw == (6, 7, 8)


def test_split_value_and_boundary_layers():
    """split_value renders a --tensor-split as layer counts per card, and
    boundary_layers reads back the last layer index each card but the
    last one holds under those counts, returning a layer index rather than
    a position within counts when entries do not start at zero."""
    entries = tuple(range(9))
    assert balance.split_value((5, 4)) == "5,4"
    assert balance.boundary_layers(entries, (5, 4)) == (4,)
    assert balance.boundary_layers(entries, (3, 3, 3)) == (2, 5)
    assert balance.boundary_layers((6, 7, 8), (2, 1)) == (7,)


def test_counts_round_trip_through_the_placement_rule():
    """An explicit --tensor-split value maps back to the same layer counts
    it names."""
    entries = tuple(range(9))
    counts = balance.counts_for_split(
        entries, tensor_split="5,4", free_per_card=[1, 1], n_cards=2
    )
    assert counts == (5, 4)


def test_counts_follow_free_memory_when_no_split_is_set():
    """With no --tensor-split value, the layer counts follow each card's
    free memory instead of splitting evenly."""
    entries = tuple(range(10))
    counts = balance.counts_for_split(
        entries, tensor_split="", free_per_card=[3, 1], n_cards=2
    )
    assert sum(counts) == 10
    assert counts[0] > counts[1]


def test_counts_for_split_round_trips_through_distribute_itself():
    """The --tensor-split value counts_for_split renders places layers on
    the same cards placement.distribute assigns them to for that value:
    running the rendered value back through place and distribute recovers
    the counts that produced it."""
    meta = _meta(n_layers=8)
    entries = balance.gpu_entries(
        meta, settings={"n-gpu-layers": "all"}, engine="llama.cpp"
    )
    free = [3, 1]
    counts = balance.counts_for_split(
        entries, tensor_split="", free_per_card=free, n_cards=2
    )
    value = balance.split_value(counts)
    devices = pl.layer_devices(meta.n_layers, len(entries), "llama.cpp")
    placed = pl.place(
        meta.tensors, meta.n_layers, devices=devices, rules=[], overrides=[]
    )
    dist = pl.distribute(
        placed,
        free_per_card=free,
        tensor_split=value,
        split_mode="layer",
        main_gpu=0,
        engine="llama.cpp",
    )
    got = [0, 0]
    for il in entries:
        card = dist.weight_share[il].index(1.0)
        got[card] += 1
    assert tuple(got) == counts


def test_balanced_split_moves_layers_off_the_smaller_card():
    """The search moves entries off the card with less free VRAM toward the
    card with more, leaving the larger card holding more layers."""
    meta = _meta(n_layers=16)
    got = balance.balanced_split(
        meta,
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all"},
        engine="llama.cpp",
        free_bytes_per_gpu=[24 * GIB, 8 * GIB],
    )
    assert got is not None
    assert sum(got.layers_per_card) == 17
    assert got.layers_per_card[0] > got.layers_per_card[1]
    assert got.value == balance.split_value(got.layers_per_card)
    assert len(got.boundary_layers) == 1


def test_balanced_split_leaves_no_card_empty():
    """A candidate that would empty a visible card is never chosen, even
    when one card's free VRAM dwarfs the other's."""
    meta = _meta(n_layers=16)
    got = balance.balanced_split(
        meta,
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all"},
        engine="llama.cpp",
        free_bytes_per_gpu=[40 * GIB, 1 * GIB],
    )
    assert got is not None
    assert min(got.layers_per_card) >= 1


def test_balanced_split_clamps_the_starting_split_to_no_empty_card():
    """The profile's own free-memory proportion can itself leave a visible
    card with no entries; the search's first candidate is the clamped split,
    not the raw one, even where giving that card its one entry back would
    not itself be a move the search goes on to try, because doing so blows
    that card's budget rather than improving on the empty state."""
    meta = _meta(n_layers=16)
    got = balance.balanced_split(
        meta,
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all"},
        engine="llama.cpp",
        free_bytes_per_gpu=[200 * GIB, 600 * MIB],
    )
    assert got is not None
    assert got.layers_per_card == (16, 1)
    assert got.boundary_layers == (15,)


def test_balanced_split_reports_whether_it_fits():
    """fits is False when even the chosen split cannot hold every card
    within its free VRAM."""
    meta = _meta(n_layers=16)
    tight = balance.balanced_split(
        meta,
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all"},
        engine="llama.cpp",
        free_bytes_per_gpu=[2 * GIB, 2 * GIB],
    )
    assert tight is not None and tight.fits is False


def test_balanced_split_does_not_favor_a_more_lopsided_shortfall():
    """On two identical cards where no split fits, a candidate that leaves
    a card over budget scores on its minimum margin alone: two equal cards
    are not handed the far more lopsided split a margin-over-cost ratio
    would favour, and the suggestion's worst margin is no worse than the
    starting split's."""
    meta = _meta(n_layers=16)
    free = [512 * MIB, 512 * MIB]
    got = balance.balanced_split(
        meta,
        0,
        settings={"ctx-size": 65536, "n-gpu-layers": "all"},
        engine="llama.cpp",
        free_bytes_per_gpu=free,
    )
    assert got is not None
    assert got.fits is False
    assert got.layers_per_card != (4, 13)

    def worst_margin(counts):
        trial = {
            "ctx-size": 65536,
            "n-gpu-layers": "all",
            "tensor-split": balance.split_value(counts),
        }
        est = estimate_memory(
            meta, 0, settings=trial, engine="llama.cpp", free_bytes_per_gpu=free
        )
        return min(f - c.total for c, f in zip(est.cards, free, strict=True))

    assert worst_margin(got.layers_per_card) >= worst_margin(got.start_layers_per_card)


def test_no_balanced_split_for_row_none_or_one_card():
    """No balanced split is computed for a row or none split mode, or when
    only one card is visible."""
    meta = _meta(n_layers=16)
    kw = dict(
        settings={"ctx-size": 8192, "n-gpu-layers": "all", "split-mode": "row"},
        engine="llama.cpp",
        free_bytes_per_gpu=[16 * GIB, 16 * GIB],
    )
    assert balance.balanced_split(meta, 0, **kw) is None
    kw["settings"] = {**kw["settings"], "split-mode": "none"}
    assert balance.balanced_split(meta, 0, **kw) is None
    assert (
        balance.balanced_split(
            meta,
            0,
            settings={"ctx-size": 8192, "n-gpu-layers": "all"},
            engine="llama.cpp",
            free_bytes_per_gpu=[16 * GIB],
        )
        is None
    )


def test_no_balanced_split_when_fewer_layers_are_placed_than_cards_are_visible():
    """No balanced split is computed when --n-gpu-layers puts fewer entries
    on the GPU than there are visible cards to divide them between."""
    meta = _meta(n_layers=16)
    got = balance.balanced_split(
        meta,
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": 1},
        engine="llama.cpp",
        free_bytes_per_gpu=[16 * GIB, 16 * GIB],
    )
    assert got is None


def test_balanced_split_balances_capacity_not_bytes():
    """With free VRAM already asymmetric, the search still moves entries so
    the smaller card is not left carrying more of the KV growth than its
    capacity allows; the split is not the one an even byte split would
    produce, nor the one the profile's own free-memory proportion lands on,
    since evaluating both directions from that starting point finds a
    strictly better one beyond it."""
    meta = _meta(n_layers=16)
    got = balance.balanced_split(
        meta,
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all"},
        engine="llama.cpp",
        free_bytes_per_gpu=[24 * GIB, 8 * GIB],
    )
    assert got is not None
    assert got.layers_per_card == (12, 5)


def test_balanced_split_falls_back_to_margin_when_no_card_grows_a_cache():
    """With no-kv-offload set, every card's marginal cost per token is zero
    and every capacity is infinite, so the search maximizes the minimum
    margin instead and keeps moving entries toward the card with more free
    VRAM for as long as doing so still relieves the smaller card's own
    margin, down to the single entry every visible card keeps."""
    meta = _meta(n_layers=16)
    got = balance.balanced_split(
        meta,
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all", "no-kv-offload": True},
        engine="llama.cpp",
        free_bytes_per_gpu=[24 * GIB, 8 * GIB],
    )
    assert got is not None
    assert got.layers_per_card == (16, 1)


def test_search_is_bounded(monkeypatch):
    """The search prices a bounded number of candidates: with two cards and
    nine entries, every one of the eight non-empty splits is scored, seven
    of them costing two estimate calls each since the base estimate a
    candidate's totals need is the same one its marginal cost is measured
    from, plus the start's own two calls priced ahead of the loop."""
    calls = []
    real = balance.estimate_memory

    def counted(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(balance, "estimate_memory", counted)
    got = balance.balanced_split(
        _meta(n_layers=8),
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all"},
        engine="llama.cpp",
        free_bytes_per_gpu=[40 * GIB, 40 * GIB],
    )
    assert got is not None
    assert got.layers_per_card == (4, 5)
    assert len(calls) == 16


def test_three_cards_climb_as_many_moves_as_there_are_entries(monkeypatch):
    """On three cards the climb's move cap is the number of entries placed
    on cards: a peak that takes every one of those rounds to reach is
    reached exactly, one round short of it is not."""
    n = 40
    meta = _meta(n_layers=n)
    total = n + 1  # 40 block layers plus the output layer
    free = [16 * GIB] * 3
    margins = {}
    # capacity is (total - a) + c: moving an entry off card 0, or moving one
    # from card 1 onto card 2, always strictly improves it, so the climb
    # drains card 0 to 1 first and then card 1 to 1, one entry per round,
    # taking every round the cap allows to reach (1, 1, total - 2).
    for a in range(1, total - 1):
        for c in range(1, total - a):
            b_ = total - a - c
            if b_ < 1:
                continue
            score = (total - a) + c
            margins[(a, b_, c)] = (score, score, score)
    costs = {k: (1, 1, 1) for k in margins}
    fake = _tabled_estimate(
        margins, costs, base_ctx=8192, free=free, default_margin=1, default_cost=1
    )
    monkeypatch.setattr(balance, "estimate_memory", fake)
    b = balance.balanced_split(
        meta, 0, **_kwargs(free=free, **{"tensor-split": "4,36,1"})
    )
    assert b.layers_per_card == (1, 1, total - 2)


def test_balanced_split_reaches_the_peak_of_a_mixed_cost_landscape():
    """A card that starts out holding only window-capped layers, carrying no
    marginal cost, is not treated as the bottleneck just because it has no
    cache growth to divide by: the search moves the boundary to the point
    of most even capacity along it, whichever side of the starting split
    that lies."""
    n_layers = 16
    pattern = tuple(i < 8 for i in range(n_layers))
    meta = _meta(n_layers=n_layers, swa=1, sliding_window_pattern=pattern)
    got = balance.balanced_split(
        meta,
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all", "tensor-split": "8,9"},
        engine="llama.cpp",
        free_bytes_per_gpu=[16 * GIB, 16 * GIB],
    )
    assert got is not None
    assert got.layers_per_card == (11, 6)


def test_balanced_split_gives_a_zero_cost_card_no_finite_capacity():
    """A card carrying no marginal cost scores as unconstrained, not as
    zero capacity: on a free-memory split that starts card 0 holding only
    window-capped layers, the search leaves most of the model on that card
    rather than draining it down, since a card whose cache does not grow
    with the context is left out of the capacity minimum."""
    n_layers = 16
    pattern = tuple(i < 8 for i in range(n_layers))
    meta = _meta(n_layers=n_layers, swa=1, sliding_window_pattern=pattern)
    got = balance.balanced_split(
        meta,
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all"},
        engine="llama.cpp",
        free_bytes_per_gpu=[12 * GIB, 20 * GIB],
    )
    assert got is not None
    assert got.layers_per_card == (10, 7)


def test_balanced_split_moves_the_boundary_to_the_better_scoring_side(monkeypatch):
    """From the profile's own split, the neighbour toward the card with more
    capacity scores worse, while the neighbour the other way, toward the
    card with less, scores better: the search takes the improving one
    regardless of which side of the boundary it falls on."""
    meta = _meta(n_layers=7)
    free = [10_000_000, 10_000_000]
    margins = {
        (5, 3): (1000, 3000),
        (6, 2): (900, 3800),
        (4, 4): (1200, 2500),
    }
    costs = {
        (5, 3): (10, 10),
        (6, 2): (10, 10),
        (4, 4): (10, 10),
    }
    monkeypatch.setattr(
        balance,
        "estimate_memory",
        _tabled_estimate(
            margins,
            costs,
            base_ctx=8192,
            free=free,
            default_margin=(-(10**9), -(10**9)),
            default_cost=(1, 1),
        ),
    )
    got = balance.balanced_split(
        meta,
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all", "tensor-split": "5,3"},
        engine="llama.cpp",
        free_bytes_per_gpu=free,
    )
    assert got is not None
    assert got.layers_per_card == (4, 4)


def test_balanced_split_takes_the_best_neighbour_not_the_first_improving_one(
    monkeypatch,
):
    """With three cards and two boundaries, a round with more than one
    improving move takes the one that scores best overall, not whichever
    boundary happens to be checked first."""
    meta = _meta(n_layers=8)
    free = [10_000_000, 10_000_000, 10_000_000]
    margins = {
        (3, 3, 3): (1000, 1000, 1000),
        (2, 4, 3): (1001, 5000, 5000),
        (3, 4, 2): (2000, 2000, 1500),
    }
    costs = {
        (3, 3, 3): (10, 10, 10),
        (2, 4, 3): (10, 10, 10),
        (3, 4, 2): (10, 10, 10),
    }
    monkeypatch.setattr(
        balance,
        "estimate_memory",
        _tabled_estimate(
            margins,
            costs,
            base_ctx=8192,
            free=free,
            default_margin=(-(10**9),) * 3,
            default_cost=(1, 1, 1),
        ),
    )
    got = balance.balanced_split(
        meta,
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all", "tensor-split": "3,3,3"},
        engine="llama.cpp",
        free_bytes_per_gpu=free,
    )
    assert got is not None
    assert got.layers_per_card == (3, 4, 2)


def test_two_cards_skip_an_unknowable_candidate_and_use_the_best_of_the_rest(
    monkeypatch,
):
    """A split whose estimate is unknowable is treated as absent, not as an
    error: scoring every other two-card split still finds the best-scoring
    one among them."""
    meta = _meta(n_layers=8)
    free = [16 * GIB, 12 * GIB]
    margins = {
        (5, 4): (10, 10),
        (4, 5): (12, 12),
        (6, 3): (9, 9),
        (3, 6): (8, 8),
        (2, 7): (20, 20),
        (1, 8): (5, 5),
        (7, 2): (4, 4),
    }
    costs = {k: (1, 1) for k in margins}
    tabled = _tabled_estimate(
        margins, costs, base_ctx=8192, free=free, default_margin=1, default_cost=1
    )

    def fake(meta, weights_bytes, *, settings, engine, **_kwargs):
        if settings.get("tensor-split") == "2,7":
            return None
        return tabled(meta, weights_bytes, settings=settings, engine=engine, **_kwargs)

    monkeypatch.setattr(balance, "estimate_memory", fake)
    b = balance.balanced_split(meta, 0, **_kwargs(free=free, **{"tensor-split": "5,4"}))
    assert b.layers_per_card == (4, 5)


def test_three_cards_skip_an_unknowable_neighbour_and_use_the_best_of_the_rest(
    monkeypatch,
):
    """A neighbour whose estimate is unknowable is treated as absent, not as
    an error: the climb keeps evaluating the remaining neighbours in the
    round and takes the best-scoring one among them."""
    meta = _meta(n_layers=8)
    free = [10_000_000, 10_000_000, 10_000_000]
    margins = {
        (3, 3, 3): (1000, 1000, 1000),
        (2, 4, 3): (1001, 5000, 5000),
        (3, 4, 2): (2000, 2000, 1500),
    }
    costs = {
        (3, 3, 3): (10, 10, 10),
        (2, 4, 3): (10, 10, 10),
        (3, 4, 2): (10, 10, 10),
    }
    tabled = _tabled_estimate(
        margins,
        costs,
        base_ctx=8192,
        free=free,
        default_margin=(-(10**9),) * 3,
        default_cost=(1, 1, 1),
    )

    def fake(meta, weights_bytes, *, settings, engine, **_kwargs):
        if settings.get("tensor-split") == "3,4,2":
            return None
        return tabled(meta, weights_bytes, settings=settings, engine=engine, **_kwargs)

    monkeypatch.setattr(balance, "estimate_memory", fake)
    got = balance.balanced_split(
        meta,
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all", "tensor-split": "3,3,3"},
        engine="llama.cpp",
        free_bytes_per_gpu=free,
    )
    assert got is not None
    assert got.layers_per_card == (2, 4, 3)


def test_balanced_split_rejects_a_move_that_only_ties_the_score(monkeypatch):
    """A candidate whose score exactly matches the current one is not
    accepted: the search keeps the profile's own split rather than marching
    across a whole chain of states that each improve nothing over the last."""
    meta = _meta(n_layers=20)
    free = [1_000_020_000, 1_000_010_000]
    margin = (2000, 1000)
    cost = (20, 10)
    base_ctx = 8192

    def fake(meta, weights_bytes, *, settings, engine, **_kwargs):
        ctx = int(settings.get("ctx-size", base_ctx) or base_ctx)
        cards = tuple(
            CardEstimate(
                weights=0,
                kv=(free[i] - margin[i]) + max(0, ctx - base_ctx) * cost[i],
                compute=0,
                overhead=0,
                state=0,
            )
            for i in range(len(free))
        )
        return MemoryEstimate(
            cards=cards,
            ram=RamEstimate(weights=0, kv=0, host=0),
            ctx=ctx,
            kv_upper_bound=False,
        )

    monkeypatch.setattr(balance, "estimate_memory", fake)
    got = balance.balanced_split(
        meta,
        0,
        settings={"ctx-size": base_ctx, "n-gpu-layers": "all", "tensor-split": "18,3"},
        engine="llama.cpp",
        free_bytes_per_gpu=free,
    )
    assert got is not None
    assert got.layers_per_card == (18, 3)


def test_three_cards_reject_a_move_that_only_ties_the_score(monkeypatch):
    """On three cards, the climb also keeps the profile's own split rather
    than taking a neighbour whose score exactly matches the current one."""
    meta = _meta(n_layers=20)
    free = [1_000_020_000, 1_000_010_000, 1_000_010_000]
    margin = (2000, 1000, 1000)
    cost = (20, 10, 10)
    base_ctx = 8192

    def fake(meta, weights_bytes, *, settings, engine, **_kwargs):
        ctx = int(settings.get("ctx-size", base_ctx) or base_ctx)
        cards = tuple(
            CardEstimate(
                weights=0,
                kv=(free[i] - margin[i]) + max(0, ctx - base_ctx) * cost[i],
                compute=0,
                overhead=0,
                state=0,
            )
            for i in range(len(free))
        )
        return MemoryEstimate(
            cards=cards,
            ram=RamEstimate(weights=0, kv=0, host=0),
            ctx=ctx,
            kv_upper_bound=False,
        )

    monkeypatch.setattr(balance, "estimate_memory", fake)
    got = balance.balanced_split(
        meta,
        0,
        settings={
            "ctx-size": base_ctx,
            "n-gpu-layers": "all",
            "tensor-split": "10,7,4",
        },
        engine="llama.cpp",
        free_bytes_per_gpu=free,
    )
    assert got is not None
    assert got.layers_per_card == (10, 7, 4)


def test_balanced_split_never_empties_a_card_even_when_the_score_would_improve(
    monkeypatch,
):
    """A move that would leave a visible card with no entries is never taken,
    even where taking it would score better than staying put."""
    meta = _meta(n_layers=6)
    free = [10_000_000, 10_000_000]
    margins = {(6, 1): (1000, 1000), (7, 0): (1200, 5000)}
    costs = {(6, 1): (10, 1), (7, 0): (10, 0)}
    monkeypatch.setattr(
        balance,
        "estimate_memory",
        _tabled_estimate(
            margins,
            costs,
            base_ctx=8192,
            free=free,
            default_margin=(-(10**9), -(10**9)),
            default_cost=(1, 1),
        ),
    )
    got = balance.balanced_split(
        meta,
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all", "tensor-split": "6,1"},
        engine="llama.cpp",
        free_bytes_per_gpu=free,
    )
    assert got is not None
    assert min(got.layers_per_card) >= 1


def test_two_cards_return_the_global_best_across_two_peaks(monkeypatch):
    """Scoring every two-card candidate reaches the higher of two peaks
    even when the hill from the start split climbs to the lower one."""
    meta = _meta(n_layers=8)
    free = [16 * GIB, 12 * GIB]
    # 8 block layers plus the output layer make 9 entries. capacities by
    # counts: start (5,4)=10, its neighbours 12 and 9, then a dip at
    # (3,6)=8 and the true peak at (2,7)=20 beyond it.
    margins = {
        (5, 4): (10, 10),
        (4, 5): (12, 12),
        (6, 3): (9, 9),
        (3, 6): (8, 8),
        (2, 7): (20, 20),
        (1, 8): (5, 5),
        (7, 2): (4, 4),
    }
    costs = {k: (1, 1) for k in margins}
    fake = _tabled_estimate(
        margins, costs, base_ctx=8192, free=free, default_margin=1, default_cost=1
    )
    monkeypatch.setattr(balance, "estimate_memory", fake)
    b = balance.balanced_split(meta, 0, **_kwargs(free=free, **{"tensor-split": "5,4"}))
    assert b.layers_per_card == (2, 7)


def test_marginal_bytes_per_token_has_no_step_parameter():
    """The step used to sample the far context is always STEP_TOKENS, not a
    caller-supplied value."""
    import inspect

    assert "step" not in inspect.signature(balance.marginal_bytes_per_token).parameters


def test_at_least_one_fills_each_empty_card_from_the_fullest():
    """Every card that starts with no entries is given one, taken from
    whichever card currently holds the most."""
    assert balance._at_least_one((5, 0, 0)) == (3, 1, 1)
    assert balance._at_least_one((2, 3)) == (2, 3)
