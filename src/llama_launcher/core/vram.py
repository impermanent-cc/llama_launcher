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


def kv_side_bytes(n_layers, n_head_kv, head_dim, ctx, quant="f16") -> int:
    """Bytes one side of the KV cache (the keys or the values) takes: one
    entry per layer, context position and KV head at that side's own head
    size and cache type."""
    per = int(n_layers) * int(ctx) * int(n_head_kv) * int(head_dim)
    return int(per * bytes_per_elem(quant))


def kv_cache_bytes(
    n_layers, n_head_kv, head_dim, ctx, k_quant="f16", v_quant="f16"
) -> int:
    return kv_side_bytes(n_layers, n_head_kv, head_dim, ctx, k_quant) + kv_side_bytes(
        n_layers, n_head_kv, head_dim, ctx, v_quant
    )


def kv_layer_mask(meta, n_layers: int) -> tuple:
    """Per layer, whether it holds a KV cache: every layer unless a per-layer
    KV head count names it (zero means none), or a full-attention interval
    does (layer i holds one when i + 1 is a multiple of it). The interval
    counts only on a header that also carries recurrent-state sizes, so a
    header naming an interval alone leaves no layer uncharged. A header with
    recurrent-state sizes and no attention heads is purely recurrent: no
    layer holds a cache."""
    n = int(n_layers)
    heads = getattr(meta, "kv_layer_heads", None)
    if heads:
        return tuple(bool(int(heads[i])) if i < len(heads) else True for i in range(n))
    interval = getattr(meta, "full_attention_interval", None)
    if interval and int(interval) > 1 and recurrent_state_bytes(meta):
        k = int(interval)
        return tuple((i + 1) % k == 0 for i in range(n))
    if recurrent_state_bytes(meta) and not int(getattr(meta, "n_head", None) or 0):
        return tuple(False for _ in range(n))
    return tuple(True for _ in range(n))


def recurrent_layer_mask(meta, kv_mask) -> tuple:
    """Per layer, whether it holds recurrent state: a layer the KV mask
    leaves uncached on a header carrying state sizes, minus the layers a
    per-layer feed-forward width array gives a non-zero width, which are
    MLP-only rather than recurrent."""
    if not recurrent_state_bytes(meta):
        return tuple(False for _ in kv_mask)
    ff = getattr(meta, "ff_layers", None)
    return tuple(
        not cached and not (ff and il < len(ff) and int(ff[il]))
        for il, cached in enumerate(kv_mask)
    )


def recurrent_state_bytes(meta) -> int:
    """f32 bytes of one recurrent layer's state for one request slot: the
    convolution state, (kernel - 1) x (inner + 2 x groups x state), plus the
    state matrix, state x inner. Zero when the header carries no ssm
    sizes."""
    conv = getattr(meta, "ssm_conv_kernel", None)
    inner = getattr(meta, "ssm_inner_size", None)
    state = getattr(meta, "ssm_state_size", None)
    groups = getattr(meta, "ssm_group_count", None) or 1
    if not (conv and inner and state):
        return 0
    conv_state = (int(conv) - 1) * (int(inner) + 2 * int(groups) * int(state))
    return _F32 * (conv_state + int(state) * int(inner))


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
    string or a bool where a number belongs, and a bool is not a slot count.
    Zero is rejected, unlike _non_negative_int."""
    return _non_negative_int(value) or None


def _non_negative_int(value) -> int | None:
    """A settings value as an int of zero or more, or None. Rejects a bool
    and anything unparsable the way the placement parser does, and rejects a
    negative number, but keeps zero, which is a meaningful count rather than
    an absent one."""
    n = _pl.int_or_none(value)
    return n if n is not None and n >= 0 else None


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
# attention, the recurrent term applies to a header carrying an inner size,
# and the host term is a multiplier on the whole per-card formula, without
# the logits term, for the host-side compute buffer.
# Calibrated against llama.cpp's memory breakdown as VRAM.md describes.
COMPUTE_TERMS = {
    "logits": 0.5,
    "ffn": 1.0,
    "residual": 2.0,
    "ssm": 64.0,
    "attn_scores": 1.0,
    "host": 0.75,
}
# Multiplier on every compute figure, the card buffers and the host buffer,
# for an engine whose graphs reserve a different amount than mainline
# llama.cpp does for the same model; an engine the table does not name
# reserves what mainline does.
ENGINE_COMPUTE_SCALE = {"ik_llama.cpp": 0.7}
# Backend context and allocator pool per visible card.
CARD_OVERHEAD_BYTES = 512 * 1024 * 1024
_F32 = 4


@dataclass(frozen=True)
class CardEstimate:
    """Card side of an estimate. `state` is the recurrent state of the
    layers the card holds, at one copy per request slot; the context
    checkpoints of those layers live in host memory and are counted in
    RamEstimate."""

    weights: int
    kv: int
    compute: int
    overhead: int
    state: int = 0

    @property
    def total(self) -> int:
        return self.weights + self.kv + self.compute + self.overhead + self.state


@dataclass(frozen=True)
class RamEstimate:
    """RAM side of an estimate. `host` is the host compute buffer at the
    engine's scale, plus the logits activation of an output layer that stayed
    in RAM, which the host computes inside that buffer. `output` is the
    buffer the server hands back, vocabulary times four bytes per slot.
    `checkpoints` is the context checkpoints of every recurrent layer,
    wherever the layer itself sits, since the server keeps them as host
    vectors."""

    weights: int
    kv: int
    host: int
    output: int = 0
    state: int = 0
    checkpoints: int = 0

    @property
    def buffers(self) -> int:
        """Host compute and output buffer together, the one RAM figure the
        readouts and the JSON breakdown show."""
        return self.host + self.output

    @property
    def total(self) -> int:
        return self.weights + self.kv + self.buffers + self.state + self.checkpoints


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
        """Weights, KV, recurrent state and checkpoints on every device, the
        figure a pool or router readout spreads across its budget."""
        return sum(c.weights + c.kv + c.state for c in self.cards) + (
            self.ram.weights + self.ram.kv + self.ram.state + self.ram.checkpoints
        )


def slot_count(settings, engine) -> int:
    """Request slots the server serves: the --parallel setting when it is a
    positive number, else the engine default of four on llama.cpp and one on
    ik_llama.cpp."""
    n = _positive_int(settings.get("parallel"))
    if n:
        return n
    return 1 if engine == "ik_llama.cpp" else 4


def _ssm_inner(meta) -> int:
    """The header's recurrent inner size, zero when it carries none."""
    return int(getattr(meta, "ssm_inner_size", None) or 0)


def host_compute_bytes(meta, *, ubatch: int, ctx: int, flash_attn: bool) -> int:
    """The host-side compute buffer: the host term of the table times the
    per-card formula without its logits term, since the host reserves a
    card-sized buffer for the same graph rather than one that follows the
    embedding size alone."""
    base = compute_bytes(
        meta, ubatch=ubatch, ctx=ctx, flash_attn=flash_attn, logits=False
    )
    return int(COMPUTE_TERMS["host"] * base)


def engine_scaled(value: int, engine: str) -> int:
    """A compute figure at the engine's scale: what mainline llama.cpp
    reserves times the engine's constant, one for an engine the table does
    not name."""
    return int(int(value) * ENGINE_COMPUTE_SCALE.get(engine, 1.0))


def logits_bytes(meta, ubatch: int) -> int:
    """The logits activation of one micro-batch: the table's logits term of
    an f32 entry per token and vocabulary entry. It is charged to the one
    card holding the output layer, or to the host buffer when that layer
    stays in RAM."""
    n_vocab = int(meta.n_vocab) if meta.n_vocab else 0
    return int(_F32 * int(ubatch) * COMPUTE_TERMS["logits"] * n_vocab)


def compute_bytes(
    meta, *, ubatch: int, ctx: int, flash_attn: bool, logits: bool = True
) -> int:
    """The per-card compute buffer: f32 activations for one micro-batch,
    FFN (active experts on a MoE model), residual stream, the recurrent
    activations of a header carrying an inner size and, without flash
    attention, the attention scores over the context. The logits term is
    added only when `logits` is set, since it is charged to the one card
    that holds the output layer."""
    n_ff_exp = int(meta.n_ff_exp) if meta.n_ff_exp else 0
    n_expert_used = int(meta.n_expert_used) if meta.n_expert_used else 0
    n_ff = (
        n_ff_exp * n_expert_used if n_ff_exp and n_expert_used else int(meta.n_ff or 0)
    )
    n_embd = int(meta.n_embd) if meta.n_embd else 0
    n_head = int(meta.n_head) if meta.n_head else 0
    per_token = (
        COMPUTE_TERMS["ffn"] * n_ff
        + COMPUTE_TERMS["residual"] * n_embd
        + COMPUTE_TERMS["ssm"] * _ssm_inner(meta)
    )
    scores = 0.0 if flash_attn else COMPUTE_TERMS["attn_scores"] * n_head * int(ctx)
    total = int(_F32 * int(ubatch) * (per_token + scores))
    if logits:
        total += logits_bytes(meta, ubatch)
    return total


def _flag_on(settings, engine, key) -> bool:
    setting = CATALOG.get(key)
    return bool(setting) and accepts(setting, engine) and bool(settings.get(key))


def _checkpoint_count(settings) -> int:
    """Context checkpoints kept per slot: --ctx-checkpoints, or the catalog
    default the server itself uses when the setting is unset or unusable.
    Zero is a legal value and keeps no checkpoints at all."""
    n = _non_negative_int(settings.get("ctx-checkpoints"))
    return n if n is not None else int(CATALOG["ctx-checkpoints"].default)


@dataclass(frozen=True)
class _Part:
    """One model's contribution to the estimate: per-card lists and the RAM
    totals, with the placement and distribution that produced them. The
    checkpoints of every recurrent layer are a RAM total alone, since the
    server keeps them as host vectors wherever the layer sits."""

    weights: list
    kv: list
    state: list
    ram_weights: int
    ram_kv: int
    ram_state: int
    ram_checkpoints: int
    placed: object
    dist: object


def _model_part(meta, weights_bytes, *, settings, engine, free, ctx, draft):
    """Per-card weights, KV and recurrent state plus the same in RAM for one
    model, with the checkpoints of every recurrent layer in RAM."""
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
    head_dim_k = int(
        getattr(meta, "head_dim_k", None) or ((n_embd // n_head) if n_head else 0)
    )
    head_dim_v = int(getattr(meta, "head_dim_v", None) or head_dim_k)
    k_quant = settings.get("cache-type-k", "f16")
    v_quant = settings.get("cache-type-v", "f16")
    if draft:
        if accepts(CATALOG["cache-type-k-draft"], engine):
            k_quant = settings.get("cache-type-k-draft", k_quant)
        if accepts(CATALOG["cache-type-v-draft"], engine):
            v_quant = settings.get("cache-type-v-draft", v_quant)
    heads = n_head_kv or n_head or 1
    layer_heads = getattr(meta, "kv_layer_heads", None)

    def kv_bytes_at(il):
        """One layer's KV cache, from its own head count where the header
        carries a per-layer array and from the collapsed count otherwise."""
        n = heads
        if layer_heads and il < len(layer_heads) and int(layer_heads[il]):
            n = int(layer_heads[il])
        return kv_side_bytes(1, n, head_dim_k, ctx, k_quant) + kv_side_bytes(
            1, n, head_dim_v, ctx, v_quant
        )

    mask = kv_layer_mask(meta, n_layers)
    recurrent = recurrent_layer_mask(meta, mask)
    rs = recurrent_state_bytes(meta) * slot_count(settings, engine)
    n_ckpt = _checkpoint_count(settings)
    ram_kv = ram_state = ram_ckpt = 0
    state = [0] * n_cards
    kv_in_ram = _flag_on(settings, engine, "no-kv-offload")
    for il in range(n_layers):
        card = dist.kv_card[il]
        if mask[il]:
            if card is None or kv_in_ram:
                ram_kv += kv_bytes_at(il)
            else:
                kv[card] += kv_bytes_at(il)
        elif recurrent[il]:
            ram_ckpt += rs * n_ckpt
            if card is None or kv_in_ram:
                ram_state += rs
            else:
                state[card] += rs
    return _Part(
        weights,
        kv,
        state,
        placed.cpu_bytes,
        ram_kv,
        ram_state,
        ram_ckpt,
        placed,
        dist,
    )


def _output_card(part, used_cards, *, split_mode, main_idx):
    """The card charged the logits term: the used card with the largest
    share of the output layer, or --main-gpu under row mode, where the whole
    compute buffer is already counted once there. None when the output layer
    stays in host RAM or lands on no used card, which sends the logits to
    the RAM buffers instead. The output layer's own device flag decides,
    since a model with no tensor table places its whole fallback blob at the
    output position whenever any layer is on a card."""
    placed = part.placed
    if not used_cards or not placed.output_gpu or not placed.devices[placed.n_layers]:
        return None
    if split_mode == "row":
        return main_idx if main_idx in used_cards else None
    shares = part.dist.weight_share[part.placed.n_layers]
    if not shares:
        return None
    card = max(range(len(shares)), key=lambda i: shares[i])
    if shares[card] <= 0 or card not in used_cards:
        return None
    return card


def _charge_compute(
    part,
    meta,
    compute,
    weights,
    kv,
    *,
    ubatch,
    ctx,
    flash,
    engine,
    split_mode,
    main_idx,
):
    """Adds one model's compute buffer, at the engine's scale and without
    its logits term, to every card the model puts weights or a KV cache on
    (only --main-gpu under row mode, where the buffer counts once), charges
    the logits term to the card holding the output layer, and returns the
    logits bytes RAM carries when no card does."""
    if split_mode == "row":
        used = {main_idx} if weights[main_idx] or kv[main_idx] else set()
    else:
        used = {i for i, w in enumerate(weights) if w or kv[i]}
    base = engine_scaled(
        compute_bytes(meta, ubatch=ubatch, ctx=ctx, flash_attn=flash, logits=False),
        engine,
    )
    for i in used:
        compute[i] += base
    logits = engine_scaled(logits_bytes(meta, ubatch), engine)
    card = _output_card(part, used, split_mode=split_mode, main_idx=main_idx)
    if card is None:
        return logits
    compute[card] += logits
    return 0


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
    """Weights, KV, compute, overhead and recurrent state per card plus
    weights, KV, state, checkpoints and the host compute and output buffers
    in RAM for a profile, with the draft model placed by its own flags and
    the projector on --main-gpu unless kept off the card. None when the
    metadata cannot support an estimate."""
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
    part = _model_part(
        meta,
        weights_bytes,
        settings=eff,
        engine=engine,
        free=free,
        ctx=ctx,
        draft=False,
    )
    weights, kv = part.weights, part.kv
    state = part.state
    ram_w, ram_kv = part.ram_weights, part.ram_kv
    ram_state, ram_ckpt = part.ram_state, part.ram_checkpoints
    compute = [0] * n_cards
    split_mode = eff.get("split-mode", "layer")
    main_idx = _positive_int(eff.get("main-gpu")) or 0
    main_idx = main_idx if 0 <= main_idx < n_cards else 0
    ram_logits = _charge_compute(
        part,
        meta,
        compute,
        weights,
        kv,
        ubatch=ubatch,
        ctx=ctx,
        flash=flash,
        engine=engine,
        split_mode=split_mode,
        main_idx=main_idx,
    )
    if draft_meta is not None and draft_meta.n_layers and draft_meta.n_embd:
        dctx = (
            _positive_int(eff.get("ctx-size-draft"))
            if accepts(CATALOG["ctx-size-draft"], engine)
            else None
        )
        dpart = _model_part(
            draft_meta,
            draft_weights,
            settings=eff,
            engine=engine,
            free=free,
            ctx=dctx or ctx,
            draft=True,
        )
        dw, dkv = dpart.weights, dpart.kv
        ram_logits += _charge_compute(
            dpart,
            draft_meta,
            compute,
            dw,
            dkv,
            ubatch=ubatch,
            ctx=dctx or ctx,
            flash=flash,
            engine=engine,
            split_mode=split_mode,
            main_idx=main_idx,
        )
        for i in range(n_cards):
            weights[i] += dw[i]
            kv[i] += dkv[i]
            state[i] += dpart.state[i]
        ram_w += dpart.ram_weights
        ram_kv += dpart.ram_kv
        ram_state += dpart.ram_state
        ram_ckpt += dpart.ram_checkpoints
    if mmproj_bytes:
        if _flag_on(eff, engine, "no-mmproj-offload"):
            ram_w += int(mmproj_bytes)
        else:
            weights[main_idx] += int(mmproj_bytes)
    cards = tuple(
        CardEstimate(weights[i], kv[i], compute[i], CARD_OVERHEAD_BYTES, state=state[i])
        for i in range(n_cards)
    )
    n_vocab = int(meta.n_vocab) if meta.n_vocab else 0
    host_buffer = (
        engine_scaled(
            host_compute_bytes(meta, ubatch=ubatch, ctx=ctx, flash_attn=flash), engine
        )
        + ram_logits
    )
    output_buffer = n_vocab * _F32 * slot_count(eff, engine)
    swa = bool(meta.sliding_window) and not _flag_on(eff, engine, "swa-full")
    ram = RamEstimate(
        ram_w,
        ram_kv,
        host_buffer,
        output_buffer,
        state=ram_state,
        checkpoints=ram_ckpt,
    )
    return MemoryEstimate(cards, ram, int(ctx), swa)
