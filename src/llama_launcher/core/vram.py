from dataclasses import dataclass

from . import placement as _pl
from .settings_catalog import CATALOG, accepts

_BYTES_PER_ELEM = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.0625,
    "q5_1": 0.75,
    "q5_0": 0.6875,
    "q4_1": 0.5625,
    "q4_0": 0.5625,
    "iq4_nl": 0.5625,
}


def bytes_per_elem(quant: str) -> float:
    return _BYTES_PER_ELEM.get((quant or "").lower(), 2.0)


def kv_cache_bytes(
    n_layers, n_head_kv, head_dim, ctx, k_quant="f16", v_quant="f16"
) -> int:
    per = int(n_layers) * int(ctx) * int(n_head_kv) * int(head_dim)
    return int(per * bytes_per_elem(k_quant) + per * bytes_per_elem(v_quant))


def fits(estimate_bytes: int, free_bytes: int) -> tuple[bool, int]:
    margin = int(free_bytes) - int(estimate_bytes)
    return (margin >= 0, margin)


def available_free_bytes(
    free_bytes_per_gpu, split_mode: str = "layer", main_gpu: int = 0
) -> int:
    """VRAM budget for the fit check given how llama.cpp will place the model.

    With `split-mode none` the whole model lands on a single card (`main-gpu`),
    so only that card's free VRAM counts. Every other split mode (layer/row/
    tensor, the default) spreads the model across all visible GPUs, so the budget
    is their COMBINED free VRAM.
    """
    free = [int(b) for b in free_bytes_per_gpu]
    if not free:
        return 0
    if split_mode == "none":
        idx = main_gpu if 0 <= main_gpu < len(free) else 0
        return free[idx]
    return sum(free)


def effective_ctx_size(settings: dict, engine: str) -> int | None:
    """The context the KV estimate should use for a profile's settings.

    An explicit positive ctx-size wins. With no ctx-size and
    --kv-unified-per-slot N set on an engine that accepts the flag, llama.cpp
    sizes the shared KV pool to n_parallel * N, which is knowable only when
    --parallel is an explicit positive number; --parallel -1 leaves the slot
    count to the server, so the answer is None and the caller falls back to
    the model's own trained context.
    """
    ctx = _positive_int(settings.get("ctx-size"))
    if ctx:
        return ctx
    setting = CATALOG["kv-unified-per-slot"]
    if not accepts(setting, engine):
        return None
    per_slot = _positive_int(settings.get("kv-unified-per-slot"))
    parallel = _positive_int(settings.get("parallel"))
    if per_slot and parallel:
        return per_slot * parallel
    return None


def _positive_int(value) -> int | None:
    """A settings value as a positive int, or None. Profile JSON can carry a
    string or a bool where a number belongs, and a bool is not a slot count."""
    if value is None or isinstance(value, bool):
        return None
    try:
        n = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return n if n > 0 else None


@dataclass
class RouterFit:
    est_bytes: int
    free_bytes: int
    free_per_gpu: tuple
    fits: bool
    margin: int
    models_counted: int  # members the worst case sums (min(models-max, total))
    models_total: int


def router_fit_summary(
    member_estimates, *, models_max, free_bytes_per_gpu
) -> RouterFit | None:
    """Estimate-vs-free for a router profile's Configure-tab fit readout.

    A router has no model of its own; it keeps up to --models-max member
    models resident at once (0 = unlimited), so the worst case sums the
    models-max LARGEST per-member estimates. The budget is the combined free
    VRAM (children are placed like the default layer split). None when
    unknowable -- no usable member estimate, or no GPU info -- so callers show
    nothing rather than wrong numbers.
    """
    ests = sorted((int(e) for e in member_estimates if e and int(e) > 0), reverse=True)
    if not ests or not free_bytes_per_gpu:
        return None
    counted = len(ests) if int(models_max) <= 0 else min(int(models_max), len(ests))
    est = sum(ests[:counted])
    free = available_free_bytes(free_bytes_per_gpu)
    ok, margin = fits(est, free)
    return RouterFit(
        est_bytes=est,
        free_bytes=free,
        free_per_gpu=tuple(int(b) for b in free_bytes_per_gpu),
        fits=ok,
        margin=margin,
        models_counted=counted,
        models_total=len(ests),
    )


@dataclass
class PooledFit:
    fits: bool
    margin: int
    vram_bytes: int
    ram_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.vram_bytes + self.ram_bytes


def pooled_fit(estimate_bytes: int, donations: list[tuple[str, int]]) -> PooledFit:
    vram = sum(int(b) for kind, b in donations if kind == "vram")
    ram = sum(int(b) for kind, b in donations if kind == "ram")
    total = vram + ram
    margin = total - int(estimate_bytes)
    return PooledFit(fits=margin >= 0, margin=margin, vram_bytes=vram, ram_bytes=ram)


# Coefficients of the f32 activation terms the compute buffer holds per
# micro-batch token; the attention-scores term applies without flash
# attention. Calibrated against llama.cpp's memory breakdown as VRAM.md
# describes.
COMPUTE_TERMS = {"logits": 1.0, "ffn": 3.0, "residual": 8.0, "attn_scores": 1.0}
# Backend context and allocator pool per visible card.
CARD_OVERHEAD_BYTES = 512 * 1024 * 1024
_F32 = 4


@dataclass(frozen=True)
class CardEstimate:
    weights: int
    kv: int
    compute: int
    overhead: int

    @property
    def total(self) -> int:
        return self.weights + self.kv + self.compute + self.overhead


@dataclass(frozen=True)
class RamEstimate:
    weights: int
    kv: int
    buffers: int

    @property
    def total(self) -> int:
        return self.weights + self.kv + self.buffers


@dataclass(frozen=True)
class MemoryEstimate:
    cards: tuple
    ram: RamEstimate
    ctx: int
    kv_upper_bound: bool

    @property
    def gpu_total(self) -> int:
        return sum(c.total for c in self.cards)

    @property
    def model_total(self) -> int:
        """Weights and KV on every device, the figure a pool or router
        readout spreads across its budget."""
        return (
            sum(c.weights + c.kv for c in self.cards) + self.ram.weights + self.ram.kv
        )


def compute_bytes(meta, *, ubatch: int, ctx: int, flash_attn: bool) -> int:
    """The per-card compute buffer: f32 activations for one micro-batch,
    logits, FFN (active experts on a MoE model), residual stream and,
    without flash attention, the attention scores over the context."""
    n_ff_exp = int(meta.n_ff_exp) if meta.n_ff_exp else 0
    n_expert_used = int(meta.n_expert_used) if meta.n_expert_used else 0
    n_ff = (
        n_ff_exp * n_expert_used if n_ff_exp and n_expert_used else int(meta.n_ff or 0)
    )
    n_vocab = int(meta.n_vocab) if meta.n_vocab else 0
    n_embd = int(meta.n_embd) if meta.n_embd else 0
    n_head = int(meta.n_head) if meta.n_head else 0
    per_token = (
        COMPUTE_TERMS["logits"] * n_vocab
        + COMPUTE_TERMS["ffn"] * n_ff
        + COMPUTE_TERMS["residual"] * n_embd
    )
    scores = 0.0 if flash_attn else COMPUTE_TERMS["attn_scores"] * n_head * int(ctx)
    return int(_F32 * int(ubatch) * (per_token + scores))


def _flag_on(settings, engine, key) -> bool:
    setting = CATALOG.get(key)
    return bool(setting) and accepts(setting, engine) and bool(settings.get(key))


def _model_part(meta, weights_bytes, *, settings, engine, free, ctx, draft):
    """Per-card (weights, kv), RAM weights and RAM KV for one model."""
    n_layers = int(meta.n_layers)
    ngl_key = "spec-draft-ngl" if draft else "n-gpu-layers"
    ngl_value = (
        settings.get(ngl_key, "auto") if accepts(CATALOG[ngl_key], engine) else "auto"
    )
    devices = _pl.layer_devices(
        n_layers, _pl.gpu_layer_count(ngl_value, n_layers, engine), engine
    )
    ot_key = "spec-draft-override-tensor" if draft else "override-tensor"
    overrides = (
        _pl.parse_overrides(settings.get(ot_key))
        if accepts(CATALOG[ot_key], engine)
        else []
    )
    placed = _pl.place(
        meta.tensors,
        n_layers,
        devices=devices,
        rules=_pl.cpu_rules(settings, engine, draft=draft),
        overrides=overrides,
        weights_fallback=weights_bytes or 0,
    )
    dist = _pl.distribute(
        placed,
        free_per_card=free,
        tensor_split=settings.get("tensor-split", ""),
        split_mode=settings.get("split-mode", "layer"),
        main_gpu=settings.get("main-gpu", 0),
        engine=engine,
    )
    n_cards = len(free)
    weights = [0] * n_cards
    kv = [0] * n_cards
    layer_bytes = [*list(placed.layer_gpu), placed.output_gpu]
    split_mode = settings.get("split-mode", "layer")
    if not meta.tensors and placed.output_gpu and n_cards > 1 and split_mode != "none":
        # With no tensor table the whole model is one fallback blob; spread it
        # by the same proportions a real per-layer table would land on, not
        # onto the single card the last discrete layer position would pick.
        points = _pl.split_fractions(settings.get("tensor-split", ""), free, n_cards)
        shares = [points[0]] + [points[i] - points[i - 1] for i in range(1, n_cards)]
        for card, frac in enumerate(shares):
            weights[card] += int(placed.output_gpu * frac)
    else:
        for il, share in enumerate(dist.weight_share):
            for card, frac in enumerate(share):
                weights[card] += int(layer_bytes[il] * frac)
    n_embd = int(meta.n_embd) if meta.n_embd else 0
    n_head = int(meta.n_head) if meta.n_head else 0
    n_head_kv = int(meta.n_head_kv) if meta.n_head_kv else n_head
    head_dim = (n_embd // n_head) if n_head else 0
    k_quant = settings.get("cache-type-k", "f16")
    v_quant = settings.get("cache-type-v", "f16")
    if draft:
        if accepts(CATALOG["cache-type-k-draft"], engine):
            k_quant = settings.get("cache-type-k-draft", k_quant)
        if accepts(CATALOG["cache-type-v-draft"], engine):
            v_quant = settings.get("cache-type-v-draft", v_quant)
    per_layer_kv = kv_cache_bytes(
        1,
        n_head_kv or n_head or 1,
        head_dim,
        ctx,
        k_quant,
        v_quant,
    )
    ram_kv = 0
    kv_in_ram = _flag_on(settings, engine, "no-kv-offload")
    for il in range(n_layers):
        card = dist.kv_card[il]
        if card is None or kv_in_ram:
            ram_kv += per_layer_kv
        else:
            kv[card] += per_layer_kv
    return weights, kv, placed.cpu_bytes, ram_kv


def estimate_memory(
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
    """Weights, KV, compute and overhead per card plus weights, KV and host
    buffers in RAM for a profile, with the draft model placed by its own
    flags and the projector on --main-gpu unless kept off the card. None
    when the metadata cannot support an estimate."""
    if meta is None or not meta.n_layers or not meta.n_embd:
        return None
    eff = _pl.effective_settings(settings, raw_args)
    free = [int(b) for b in free_bytes_per_gpu] or [0]
    n_cards = len(free)
    ctx = effective_ctx_size(eff, engine) or meta.ctx_train or 4096
    batch = _positive_int(eff.get("batch-size")) or int(CATALOG["batch-size"].default)
    ubatch = _positive_int(eff.get("ubatch-size")) or int(
        CATALOG["ubatch-size"].default
    )
    batch = min(batch, ctx)
    ubatch = min(ubatch, batch)
    flash = str(eff.get("flash-attn", "auto")) != "off"
    weights, kv, ram_w, ram_kv = _model_part(
        meta,
        weights_bytes,
        settings=eff,
        engine=engine,
        free=free,
        ctx=ctx,
        draft=False,
    )
    compute = [0] * n_cards
    split_mode = eff.get("split-mode", "layer")
    main_idx = _positive_int(eff.get("main-gpu")) or 0
    main_idx = main_idx if 0 <= main_idx < n_cards else 0
    if split_mode == "row":
        used_cards = {main_idx} if weights[main_idx] or kv[main_idx] else set()
    else:
        used_cards = {i for i in range(n_cards) if weights[i] or kv[i]}
    for i in used_cards:
        compute[i] += compute_bytes(meta, ubatch=ubatch, ctx=ctx, flash_attn=flash)
    if draft_meta is not None and draft_meta.n_layers and draft_meta.n_embd:
        dctx = (
            _positive_int(eff.get("ctx-size-draft"))
            if accepts(CATALOG["ctx-size-draft"], engine)
            else None
        )
        dw, dkv, dram_w, dram_kv = _model_part(
            draft_meta,
            draft_weights,
            settings=eff,
            engine=engine,
            free=free,
            ctx=dctx or ctx,
            draft=True,
        )
        for i in range(n_cards):
            weights[i] += dw[i]
            kv[i] += dkv[i]
        if split_mode == "row":
            draft_used = {main_idx} if dw[main_idx] or dkv[main_idx] else set()
        else:
            draft_used = {i for i in range(n_cards) if dw[i] or dkv[i]}
        for i in draft_used:
            compute[i] += compute_bytes(
                draft_meta, ubatch=ubatch, ctx=dctx or ctx, flash_attn=flash
            )
        ram_w += dram_w
        ram_kv += dram_kv
    if mmproj_bytes:
        if _flag_on(eff, engine, "no-mmproj-offload"):
            ram_w += int(mmproj_bytes)
        else:
            weights[main_idx] += int(mmproj_bytes)
    cards = tuple(
        CardEstimate(weights[i], kv[i], compute[i], CARD_OVERHEAD_BYTES)
        for i in range(n_cards)
    )
    n_vocab = int(meta.n_vocab) if meta.n_vocab else 0
    buffers = int(n_vocab * batch * _F32)
    swa = bool(meta.sliding_window) and not _flag_on(eff, engine, "swa-full")
    return MemoryEstimate(cards, RamEstimate(ram_w, ram_kv, buffers), int(ctx), swa)
