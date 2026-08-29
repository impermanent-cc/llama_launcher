from dataclasses import dataclass

_BYTES_PER_ELEM = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q8_0": 1.0625, "q5_1": 0.75, "q5_0": 0.6875,
    "q4_1": 0.5625, "q4_0": 0.5625, "iq4_nl": 0.5625,
}


def bytes_per_elem(quant: str) -> float:
    return _BYTES_PER_ELEM.get((quant or "").lower(), 2.0)


@dataclass
class VramEstimate:
    kv_bytes: int
    weights_bytes: int
    overhead_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.kv_bytes + self.weights_bytes + self.overhead_bytes


def kv_cache_bytes(n_layers, n_head_kv, head_dim, ctx, k_quant="f16", v_quant="f16") -> int:
    per = int(n_layers) * int(ctx) * int(n_head_kv) * int(head_dim)
    return int(per * bytes_per_elem(k_quant) + per * bytes_per_elem(v_quant))


def estimate(*, n_layers, n_head, n_head_kv, n_embd, ctx,
             k_quant="f16", v_quant="f16", weights_bytes=0,
             overhead_bytes=536870912) -> VramEstimate:
    head_dim = (int(n_embd) // int(n_head)) if n_head else 0
    kv = kv_cache_bytes(n_layers, n_head_kv, head_dim, ctx, k_quant, v_quant)
    return VramEstimate(kv_bytes=kv, weights_bytes=int(weights_bytes),
                        overhead_bytes=int(overhead_bytes))


def estimate_for_model(meta, weights_bytes, *, ctx_size=None,
                       k_quant="f16", v_quant="f16") -> int:
    """Weights+KV total-bytes estimate for a model from its GGUF metadata.

    `meta` is a model_info.inspect_model result (duck-typed: n_layers/n_head/
    n_head_kv/n_embd/ctx_train). When metadata is too thin for a KV estimate,
    fall back to the weights size alone. Shared by the single-node preflight
    (LaunchController.vram_check) and the pooled one (Check fit) so both derive
    the estimate identically. ctx precedence: explicit ctx_size -> meta.ctx_train
    -> 4096."""
    if meta is None or not meta.n_layers or not meta.n_embd:
        return int(weights_bytes or 0)
    ctx = ctx_size or meta.ctx_train or 4096
    est = estimate(
        n_layers=meta.n_layers, n_head=meta.n_head or 1,
        n_head_kv=meta.n_head_kv or meta.n_head or 1, n_embd=meta.n_embd, ctx=ctx,
        k_quant=k_quant, v_quant=v_quant, weights_bytes=weights_bytes or 0)
    return est.total_bytes


def fits(estimate_bytes: int, free_bytes: int) -> tuple[bool, int]:
    margin = int(free_bytes) - int(estimate_bytes)
    return (margin >= 0, margin)


def available_free_bytes(free_bytes_per_gpu, split_mode: str = "layer",
                         main_gpu: int = 0) -> int:
    """VRAM budget for the fit check given how llama.cpp will place the model.

    With `split-mode none` the whole model lands on a single card (`main-gpu`),
    so only that card's free VRAM counts. Every other split mode (layer/row/
    tensor, the default) spreads the model across all visible GPUs, so the budget
    is their COMBINED free VRAM -- using a single card's free (the old behaviour)
    wrongly warned that a model fitting across two GPUs would not fit.
    """
    free = [int(b) for b in free_bytes_per_gpu]
    if not free:
        return 0
    if split_mode == "none":
        idx = main_gpu if 0 <= main_gpu < len(free) else 0
        return free[idx]
    return sum(free)


@dataclass
class FitSummary:
    est_bytes: int
    free_bytes: int
    free_per_gpu: tuple
    fits: bool
    margin: int


def fit_summary(meta, weights_bytes, *, settings: dict,
                free_bytes_per_gpu) -> FitSummary | None:
    """The single estimate-vs-free computation behind every single-node VRAM
    preflight (the launch-time vram_check dialog and the Configure tab's live
    fit readout), so they can never disagree. `settings` is a profile settings
    dict; placement (split-mode/main-gpu) picks the free-VRAM budget the same
    way llama.cpp will place the model. None when the answer is unknowable --
    metadata too thin for a KV estimate, or no GPU info -- so callers show
    nothing rather than wrong numbers.
    """
    if meta is None or not meta.n_layers or not meta.n_embd or not free_bytes_per_gpu:
        return None
    free = available_free_bytes(free_bytes_per_gpu,
                                settings.get("split-mode", "layer"),
                                settings.get("main-gpu", 0))
    est = estimate_for_model(meta, weights_bytes,
                             ctx_size=settings.get("ctx-size"),
                             k_quant=settings.get("cache-type-k", "f16"),
                             v_quant=settings.get("cache-type-v", "f16"))
    ok, margin = fits(est, free)
    return FitSummary(est_bytes=est, free_bytes=free,
                      free_per_gpu=tuple(int(b) for b in free_bytes_per_gpu),
                      fits=ok, margin=margin)


@dataclass
class RouterFit:
    est_bytes: int
    free_bytes: int
    free_per_gpu: tuple
    fits: bool
    margin: int
    models_counted: int   # members the worst case sums (min(models-max, total))
    models_total: int


def router_fit_summary(member_estimates, *, models_max,
                       free_bytes_per_gpu) -> RouterFit | None:
    """Estimate-vs-free for a router profile's Configure-tab fit readout.

    A router has no model of its own; it keeps up to --models-max member
    models resident at once (0 = unlimited), so the worst case sums the
    models-max LARGEST per-member estimates. The budget is the combined free
    VRAM (children are placed like the default layer split). None when
    unknowable -- no usable member estimate, or no GPU info -- so callers show
    nothing rather than wrong numbers.
    """
    ests = sorted((int(e) for e in member_estimates if e and int(e) > 0),
                  reverse=True)
    if not ests or not free_bytes_per_gpu:
        return None
    counted = len(ests) if int(models_max) <= 0 else min(int(models_max), len(ests))
    est = sum(ests[:counted])
    free = available_free_bytes(free_bytes_per_gpu)
    ok, margin = fits(est, free)
    return RouterFit(est_bytes=est, free_bytes=free,
                     free_per_gpu=tuple(int(b) for b in free_bytes_per_gpu),
                     fits=ok, margin=margin,
                     models_counted=counted, models_total=len(ests))


@dataclass
class PooledFit:
    fits: bool
    margin: int
    vram_bytes: int
    ram_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.vram_bytes + self.ram_bytes


def pooled_fit(estimate_bytes: int,
               donations: list[tuple[str, int]]) -> PooledFit:
    vram = sum(int(b) for kind, b in donations if kind == "vram")
    ram = sum(int(b) for kind, b in donations if kind == "ram")
    total = vram + ram
    margin = total - int(estimate_bytes)
    return PooledFit(fits=margin >= 0, margin=margin, vram_bytes=vram, ram_bytes=ram)
