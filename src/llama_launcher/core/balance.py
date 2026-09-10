"""A capacity-balanced --tensor-split: each card's marginal cost per token,
the whole-layer boundaries that can be suggested, and the search that picks
one."""

from dataclasses import dataclass

from . import placement as _pl
from .vram import estimate_memory

STEP_TOKENS = 4096
_UNSPLIT_MODES = ("row", "none")


def _estimate_at(meta, weights_bytes, *, ctx, kwargs, settings):
    return estimate_memory(
        meta, weights_bytes, settings={**settings, "ctx-size": ctx}, **kwargs
    )


def _cache_bytes(est) -> tuple:
    return tuple(c.kv + c.state for c in est.cards)


def marginal_bytes_per_token(
    meta,
    weights_bytes,
    *,
    settings,
    engine,
    free_bytes_per_gpu,
    raw_args="",
    draft_meta=None,
    draft_weights=0,
    mmproj_bytes=0,
    base=None,
):
    """Each card's KV and state total at the profile's own context,
    subtracted from the same total one STEP_TOKENS further out and divided
    by STEP_TOKENS: the recurrent-state term is per state cell and does
    not move with context, so it cancels out and what is left is the rate
    the KV cache grows by, with window caps, cache types and slot rules
    needing no second formula. Never negative. None when the estimate is
    unknowable. base, when given, is the MemoryEstimate for these same
    settings, already computed by the caller, so the profile's own context
    is not priced a second time."""
    eff = _pl.effective_settings(settings, raw_args)
    kwargs = dict(
        engine=engine,
        free_bytes_per_gpu=free_bytes_per_gpu,
        draft_meta=draft_meta,
        draft_weights=draft_weights,
        mmproj_bytes=mmproj_bytes,
    )
    if base is None:
        base = estimate_memory(meta, weights_bytes, settings=eff, **kwargs)
    if base is None:
        return None
    ctx = base.ctx
    far = _estimate_at(
        meta, weights_bytes, ctx=ctx + STEP_TOKENS, kwargs=kwargs, settings=eff
    )
    if far is None:
        return None
    lo, hi = _cache_bytes(base), _cache_bytes(far)
    return tuple(
        max(0.0, (hi_i - lo_i) / STEP_TOKENS) for lo_i, hi_i in zip(lo, hi, strict=True)
    )


def gpu_entries(meta, *, settings, engine, raw_args="") -> tuple:
    """Layer indices --n-gpu-layers puts on a card, index n_layers being the
    output layer, in ascending order: the entries the split points divide."""
    eff = _pl.effective_settings(settings, raw_args)
    n_layers = int(meta.n_layers)
    ngl = _pl.gpu_layer_count(eff.get("n-gpu-layers", "auto"), n_layers, engine)
    devices = _pl.layer_devices(n_layers, ngl, engine)
    return tuple(i for i, on_card in enumerate(devices) if on_card)


def split_value(counts) -> str:
    """A --tensor-split value as layer counts per card, the rendering whose
    boundary cannot round onto another layer."""
    return ",".join(str(int(c)) for c in counts)


def boundary_layers(entries, counts) -> tuple:
    """The last layer index each card but the last one holds. Every count
    must be positive: a zero count wraps to the last entry."""
    out = []
    acc = 0
    for c in list(counts)[:-1]:
        acc += int(c)
        out.append(entries[acc - 1])
    return tuple(out)


def counts_for_split(entries, *, tensor_split, free_per_card, n_cards) -> tuple:
    """Entries per card under a --tensor-split value, or under each card's
    free memory when it is unset, by the same rule as placement.distribute."""
    points = _pl.split_fractions(tensor_split, free_per_card, n_cards)
    act = len(entries)
    counts = [0] * n_cards
    for k in range(act):
        frac = k / act
        card = next((i for i, p in enumerate(points) if p > frac), n_cards - 1)
        counts[card] += 1
    return tuple(counts)


@dataclass(frozen=True)
class BalancedSplit:
    """A suggested --tensor-split: its value, the entries each card would
    hold, the last layer index on each card but the last, whether every
    card fits at it with no offload, and the search's starting point, the
    profile's own resolved entries per card clamped to leave no visible
    card empty, against which a caller can tell whether the suggestion
    changes anything."""

    value: str
    layers_per_card: tuple
    boundary_layers: tuple
    fits: bool
    start_layers_per_card: tuple


def _priced(counts, meta, weights_bytes, *, kwargs, settings, free, cache):
    """The score for one candidate, cached by its counts, compared
    lexicographically. Feasibility (every card fits) is scored first. Among
    feasible candidates, the next term is the minimum capacity over the
    cards whose marginal cost is non-zero, a card with zero marginal cost
    having infinite capacity and being left out of the minimum, and then
    the minimum margin, which decides where no card has a non-zero
    marginal cost. A candidate that leaves any card over budget scores on
    its minimum margin alone: a negative margin over a cost is not a
    capacity, and ranking it as one would favour the deeper shortfall.
    None when the estimate is unknowable at that split."""
    key = tuple(counts)
    if key in cache:
        return cache[key]
    trial = {**settings, "tensor-split": split_value(counts)}
    est = estimate_memory(meta, weights_bytes, settings=trial, **kwargs)
    if est is None:
        cache[key] = None
        return None
    margins = [int(f) - c.total for c, f in zip(est.cards, free, strict=True)]
    feasible = all(m >= 0 for m in margins)
    if feasible:
        costs = marginal_bytes_per_token(
            meta, weights_bytes, settings=trial, base=est, **kwargs
        ) or tuple(0.0 for _ in margins)
        caps = [
            (m / c if c > 0 else float("inf"))
            for m, c in zip(margins, costs, strict=True)
        ]
        finite = [c for c in caps if c != float("inf")]
        score = (True, min(finite) if finite else float("inf"), min(margins))
    else:
        score = (False, min(margins))
    cache[key] = score
    return score


def _at_least_one(counts) -> tuple:
    """The same counts with every card holding at least one entry, each
    missing entry taken from the card holding the most."""
    out = list(counts)
    for i, n in enumerate(out):
        if n == 0:
            donor = max(range(len(out)), key=lambda k: out[k])
            if out[donor] <= 1:
                break
            out[donor] -= 1
            out[i] = 1
    return tuple(out)


def _neighbours(counts) -> tuple:
    """Every candidate one whole-layer move away: for each boundary, the
    counts with one entry moved across it in either direction, leaving out a
    move that would empty the card it is taken from."""
    out = []
    for b in range(len(counts) - 1):
        if counts[b] > 1:
            moved = list(counts)
            moved[b] -= 1
            moved[b + 1] += 1
            out.append(tuple(moved))
        if counts[b + 1] > 1:
            moved = list(counts)
            moved[b + 1] -= 1
            moved[b] += 1
            out.append(tuple(moved))
    return tuple(out)


def balanced_split(
    meta,
    weights_bytes,
    *,
    settings,
    engine,
    free_bytes_per_gpu,
    raw_args="",
    draft_meta=None,
    draft_weights=0,
    mmproj_bytes=0,
):
    """The --tensor-split whose per-card capacity is most even. On two
    cards every whole-layer split is scored and the best-scoring one is
    returned. On three or more cards the search climbs from the profile's
    own split, each round taking whichever single whole-layer move across
    whichever boundary scores best among every move that does not empty a
    card, keeping it only while it strictly improves on the round before,
    for at most as many rounds as there are entries to place. None when
    fewer than two cards are visible, when the split mode places no
    contiguous runs (row and none), when no layer is on a card, or when the
    estimate is unknowable."""
    eff = _pl.effective_settings(settings, raw_args)
    if str(eff.get("split-mode", "layer") or "layer") in _UNSPLIT_MODES:
        return None
    free = [int(b) for b in free_bytes_per_gpu]
    n_cards = len(free)
    if n_cards < 2:
        return None
    entries = gpu_entries(meta, settings=eff, engine=engine)
    if len(entries) < n_cards:
        return None
    kwargs = dict(
        engine=engine,
        free_bytes_per_gpu=free,
        draft_meta=draft_meta,
        draft_weights=draft_weights,
        mmproj_bytes=mmproj_bytes,
    )
    cache = {}

    def priced(counts):
        return _priced(
            counts,
            meta,
            weights_bytes,
            kwargs=kwargs,
            settings=eff,
            free=free,
            cache=cache,
        )

    start = _at_least_one(
        counts_for_split(
            entries,
            tensor_split=eff.get("tensor-split", ""),
            free_per_card=free,
            n_cards=n_cards,
        )
    )
    counts, current = start, priced(start)
    if current is None:
        return None
    if n_cards == 2:
        total = len(entries)
        for cand in ((k, total - k) for k in range(1, total)):
            got = priced(cand)
            if got is not None and got > current:
                counts, current = cand, got
    else:
        for _ in range(len(entries)):
            best, best_score = None, None
            for cand in _neighbours(counts):
                got = priced(cand)
                if got is None:
                    continue
                if best_score is None or got > best_score:
                    best, best_score = cand, got
            if best is None or best_score <= current:
                break
            counts, current = best, best_score
    return BalancedSplit(
        split_value(counts),
        tuple(counts),
        boundary_layers(entries, counts),
        current[0],
        tuple(start),
    )
