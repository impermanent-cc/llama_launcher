"""Per-card and RAM verdicts for a memory estimate, the messages a launch
shows for them, and their rendering as readout lines, tooltip, dialog
text and JSON."""

import html
import re
from dataclasses import dataclass

from . import balance as _balance
from . import placement as _pl
from .settings_catalog import CATALOG, accepts
from .vram import estimate_memory

_GIB = 1024**3
_MIB = 1024**2
_FIT_TARGET_DEFAULT_MIB = 1024
_FIT_CTX_DEFAULT = 4096
_CTX_ALIGN = 256
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class CardFit:
    index: int
    est: int
    free: int
    margin: int
    fits: bool


@dataclass(frozen=True)
class RamFit:
    est: int
    available: int | None
    margin: int | None
    fits: bool | None


@dataclass(frozen=True)
class Message:
    text: str
    dialog: bool


@dataclass(frozen=True)
class FitReport:
    estimate: object
    cards: tuple
    ram: RamFit
    messages: tuple
    fits: bool
    balanced: object = None


def _gib(n) -> str:
    return f"{n / _GIB:.1f}"


def _state_parts(state: int, checkpoints: int = 0, unit: str = "") -> str:
    """The recurrent-state clause of a breakdown, with the checkpoints term
    where the device carries one (RAM alone), empty when the model has
    neither."""
    if not (state or checkpoints):
        return ""
    tail = f" {unit}" if unit else ""
    out = f", state {_gib(state)}{tail}"
    if checkpoints:
        out += f", checkpoints {_gib(checkpoints)}{tail}"
    return out


def _fit_state(settings) -> str:
    """'unset', 'on' or 'off' as the engine will see it."""
    raw = settings.get("fit", "unset")
    if raw is True:
        return "on"
    if raw is False:
        return "off"
    value = str(raw or "unset")
    return value if value in ("on", "off") else "unset"


def _fit_active_mainline(settings, engine, n_cards: int = 1) -> bool:
    """SPEC 2.22: fit is active only when nothing already claims the offload
    or the split the fit search would otherwise make: no CPU offload flag
    (--cpu-moe, --n-cpu-moe, --n-cpu-ffn), no --n-gpu-layers, no
    --override-tensor, no --tensor-split, and the split mode is layer
    (or there is only one card)."""
    if engine != "llama.cpp" or _fit_state(settings) == "off":
        return False
    ngl = settings.get("n-gpu-layers", "auto")
    if ngl not in (None, "auto"):
        return False
    if str(settings.get("override-tensor", "") or "").strip():
        return False
    if _pl.cpu_rules(settings, engine):
        return False
    if str(settings.get("tensor-split", "") or "").strip():
        return False
    split_mode = str(settings.get("split-mode", "layer") or "layer")
    return split_mode == "layer" or n_cards <= 1


def _ctx_size_set(settings) -> bool:
    """True when the effective settings carry a positive --ctx-size, the
    case where upstream fit keeps that context rather than shrinking it."""
    n = _pl.int_or_none(settings.get("ctx-size"))
    return bool(n and n > 0)


def _fit_margins(settings, n_cards) -> list:
    """Per-card --fit-target margins in bytes, the single value broadcast."""
    raw = str(settings.get("fit-target", "") or "")
    vals = []
    for tok in raw.replace("/", ",").split(","):
        tok = tok.strip()
        if tok:
            try:
                vals.append(int(float(tok)) * _MIB)
            except ValueError:
                vals = []
                break
    if not vals:
        vals = [_FIT_TARGET_DEFAULT_MIB * _MIB]
    if len(vals) == 1:
        vals = vals * n_cards
    return (vals + [vals[-1]] * n_cards)[:n_cards]


def _fit_ctx_floor(settings) -> int:
    n = _pl.int_or_none(settings.get("fit-ctx"))
    return n if n and n > 0 else _FIT_CTX_DEFAULT


def predicted_fit_ctx(
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
    """(context, drops_layers) llama.cpp's fit reaches when no --ctx-size is
    set: the largest context, rounded down to a multiple of 256 and never
    below the floor (--fit-ctx, used unrounded), whose summed card total
    fits the summed free VRAM less each card's --fit-target margin; when
    even the floor does not fit, the floor with drops_layers True. None
    when the estimate is unknowable or a positive --ctx-size is set, since
    upstream then keeps that context and moves layers to RAM instead."""
    eff = _pl.effective_settings(settings, raw_args)
    if _ctx_size_set(eff):
        return None
    kwargs = dict(
        engine=engine,
        free_bytes_per_gpu=free_bytes_per_gpu,
        draft_meta=draft_meta,
        draft_weights=draft_weights,
        mmproj_bytes=mmproj_bytes,
    )
    top = estimate_memory(meta, weights_bytes, settings=eff, **kwargs)
    if top is None:
        return None
    n_cards = len(top.cards)
    budget = sum(int(b) for b in free_bytes_per_gpu) - sum(_fit_margins(eff, n_cards))
    floor = min(_fit_ctx_floor(eff), top.ctx)
    low = estimate_memory(
        meta, weights_bytes, settings={**eff, "ctx-size": floor}, **kwargs
    )
    if low.gpu_total > budget:
        return floor, True
    if top.gpu_total <= budget or top.ctx == floor:
        return top.ctx, False
    per_token = (top.gpu_total - low.gpu_total) / (top.ctx - floor)
    ctx = (
        floor + int((budget - low.gpu_total) / per_token) if per_token > 0 else top.ctx
    )
    ctx = max(floor, ctx - ctx % _CTX_ALIGN)
    return min(ctx, top.ctx), False


def _all_fit(est, free) -> bool:
    return all(c.total <= int(f) for c, f in zip(est.cards, free, strict=False))


def is_moe(meta) -> bool:
    """A model counts as MoE when its metadata names an expert count or any
    tensor name carries an expert-weight suffix, so a table missing the
    hyperparameter still routes to the expert offload flag."""
    if meta.expert_count:
        return True
    return any("_exps" in t.name for t in (meta.tensors or ()))


def _offload_search(
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
    """(candidate settings, layer count) of the smallest CPU offload at
    which every card fits: --n-cpu-moe on a MoE model, --n-cpu-ffn on a
    dense one, or an --override-tensor alternation of the same layers
    where the engine lacks --n-cpu-ffn, appended after any override-tensor
    value the profile already carries so the search baseline matches the
    reported shortfall and the suggestion keeps the user's own rules. The
    layer count is comparable across two searches even where the settings
    are an --override-tensor string rather than a number. None when no
    count up to the layer count fits."""
    eff = _pl.effective_settings(settings, raw_args)
    n_layers = int(meta.n_layers)
    moe = is_moe(meta)
    key = "n-cpu-moe" if moe else "n-cpu-ffn"
    use_override = not accepts(CATALOG[key], engine)
    existing_override = (
        str(eff.get("override-tensor", "") or "").strip().rstrip(",").strip()
    )

    def candidate(n):
        if not use_override:
            return {key: n}
        layers = "|".join(str(i) for i in range(n))
        value = rf"blk\.({layers}){_pl.DENSE_FFN_REGEX}=CPU"
        if existing_override:
            value = f"{existing_override},{value}"
        return {"override-tensor": value}

    def fits(n):
        est = estimate_memory(
            meta,
            weights_bytes,
            settings={**eff, **candidate(n)},
            engine=engine,
            free_bytes_per_gpu=free_bytes_per_gpu,
            draft_meta=draft_meta,
            draft_weights=draft_weights,
            mmproj_bytes=mmproj_bytes,
        )
        return est is not None and _all_fit(est, free_bytes_per_gpu)

    if fits(0):
        return None
    if not fits(n_layers):
        return None
    lo, hi = 1, n_layers
    while lo < hi:
        mid = (lo + hi) // 2
        if fits(mid):
            hi = mid
        else:
            lo = mid + 1
    return candidate(lo), lo


def smallest_fitting_offload(
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
    """(key, value) of the smallest CPU offload at which every card fits.
    None when no count up to the layer count fits."""
    found = _offload_search(
        meta,
        weights_bytes,
        settings=settings,
        engine=engine,
        free_bytes_per_gpu=free_bytes_per_gpu,
        raw_args=raw_args,
        draft_meta=draft_meta,
        draft_weights=draft_weights,
        mmproj_bytes=mmproj_bytes,
    )
    if found is None:
        return None
    value, _count = found
    return next(iter(value.items()))


def _shortfall_text(cards) -> str:
    parts = [
        f"GPU{c.index}: est ~{_gib(c.est)} GiB exceeds free ~{_gib(c.free)} "
        f"GiB by ~{_gib(-c.margin)} GiB"
        for c in cards
        if not c.fits
    ]
    return "; ".join(parts) + "."


def _suggestion_text(found, split_value=None) -> str:
    """The offload-count clause of a shortfall message: the smallest count
    that fits, naming the split it was found at when that split is a
    balanced suggestion rather than the profile's own, since the count is
    minimal only in combination with that split."""
    at = f" at --tensor-split {split_value}" if split_value else ""
    if found is None:
        return (
            f" Even offloading every layer leaves the card over budget{at}, "
            "so no offload count fits; lower the context or the KV "
            "cache type."
        )
    key, value = found
    return f" Smallest offload that fits{at}: --{key} {value}."


def _ik_moe_offload(
    meta,
    weights_bytes,
    *,
    settings,
    engine,
    free_bytes_per_gpu,
    raw_args,
    draft_meta,
    draft_weights,
    mmproj_bytes,
):
    """(estimate, layer_count) once ik_llama.cpp is told to keep enough
    layers' experts in host RAM to fit every card, or the smallest offload
    that fits none, the whole model's layers with --cpu-moe. None, 0 when
    the estimate is unknowable."""
    found = smallest_fitting_offload(
        meta,
        weights_bytes,
        settings=settings,
        engine=engine,
        free_bytes_per_gpu=free_bytes_per_gpu,
        raw_args=raw_args,
        draft_meta=draft_meta,
        draft_weights=draft_weights,
        mmproj_bytes=mmproj_bytes,
    )
    n_layers = int(meta.n_layers)
    if found is not None:
        _, layers = found
        overlay = {"n-cpu-moe": layers}
    else:
        layers = n_layers
        overlay = {"cpu-moe": True}
    eff = _pl.effective_settings(settings, raw_args)
    adjusted = estimate_memory(
        meta,
        weights_bytes,
        settings={**eff, **overlay},
        engine=engine,
        free_bytes_per_gpu=free_bytes_per_gpu,
        draft_meta=draft_meta,
        draft_weights=draft_weights,
        mmproj_bytes=mmproj_bytes,
    )
    return adjusted, layers


def _balanced_text(balanced, eff, fit_active: bool) -> str:
    """The balanced-split sentence of a shortfall message: the value, the
    value it replaces where the profile sets one, and, only where --fit
    would otherwise act on the profile's own settings, the note that an
    explicit split keeps it from acting."""
    if balanced is None:
        return ""
    current = str(eff.get("tensor-split", "") or "").strip()
    fitting = "fits every card" if balanced.fits else "does not fit on its own"
    text = (
        f" Balanced --tensor-split {balanced.value} "
        f"(card 0 keeps layers up to {balanced.boundary_layers[0]}) {fitting}."
        if balanced.boundary_layers
        else f" Balanced --tensor-split {balanced.value} {fitting}."
    )
    if current:
        text += f" It replaces the profile's --tensor-split {current}."
    if fit_active:
        text += " Setting --tensor-split keeps --fit from acting."
    return text


def _ik_moe_note(layers: int) -> str:
    return (
        f"With --fit on, ik_llama.cpp keeps the experts of {layers} layer"
        f"{'s' if layers != 1 else ''} in RAM to fit the model; context and "
        "layer count stay as set and the RAM total includes them."
    )


def _messages(
    meta,
    weights_bytes,
    est,
    cards,
    ram,
    *,
    settings,
    engine,
    free_bytes_per_gpu,
    raw_args,
    draft_meta,
    draft_weights,
    mmproj_bytes,
    ik_moe_layers=None,
    balanced=None,
) -> list:
    eff = _pl.effective_settings(settings, raw_args)
    kwargs = dict(
        settings=settings,
        engine=engine,
        free_bytes_per_gpu=free_bytes_per_gpu,
        raw_args=raw_args,
        draft_meta=draft_meta,
        draft_weights=draft_weights,
        mmproj_bytes=mmproj_bytes,
    )
    out = []
    over = [c for c in cards if not c.fits]
    if ik_moe_layers is not None:
        out.append(Message(_ik_moe_note(ik_moe_layers), dialog=False))
        if over:
            out.append(
                Message(
                    _shortfall_text(over) + " Even keeping every layer's "
                    "experts in RAM does not fit.",
                    dialog=True,
                )
            )
    elif over:
        differs = balanced is not None and (
            tuple(balanced.layers_per_card) != tuple(balanced.start_layers_per_card)
        )
        profile_search = _offload_search(meta, weights_bytes, **kwargs)
        balanced_search = None
        if differs and balanced is not None and not balanced.fits:
            balanced_search = _offload_search(
                meta,
                weights_bytes,
                **{
                    **kwargs,
                    "settings": {**eff, "tensor-split": balanced.value},
                    "raw_args": "",
                },
            )
        smaller_at_balanced = balanced_search is not None and (
            profile_search is None or balanced_search[1] < profile_search[1]
        )
        name_balanced = differs and (balanced.fits or smaller_at_balanced)
        at_balanced = name_balanced and not balanced.fits
        search = balanced_search if at_balanced else profile_search
        found = next(iter(search[0].items())) if search is not None else None
        suggestion = _suggestion_text(found, balanced.value if at_balanced else None)
        fit_active = _fit_active_mainline(eff, engine, len(free_bytes_per_gpu))
        balanced_text = (
            _balanced_text(balanced, eff, fit_active) if name_balanced else ""
        )
        base = _shortfall_text(over)
        state = _fit_state(eff)
        if fit_active:
            if _ctx_size_set(eff):
                expert_note = " (experts first on a MoE model)" if is_moe(meta) else ""
                text = (
                    f"{base} With --fit active llama.cpp will keep the set "
                    f"context and move whole layers to RAM{expert_note} "
                    "rather than fail." + suggestion + balanced_text
                )
            else:
                predicted = predicted_fit_ctx(meta, weights_bytes, **kwargs)
                ctx, drops = predicted if predicted else (est.ctx, False)
                text = (
                    f"{base} With --fit active llama.cpp will shrink the context "
                    f"to about {ctx}"
                    + (" and then drop whole layers" if drops else "")
                    + " rather than fail."
                    + (
                        " To keep the context set --fit off and use the offload below."
                        if found
                        else ""
                    )
                    + suggestion
                    + balanced_text
                )
            out.append(Message(text, dialog=state == "unset"))
        elif engine == "ik_llama.cpp" and state == "on" and not is_moe(meta):
            out.append(
                Message(
                    f"{base} With --fit on, ik_llama.cpp refuses to "
                    "load a dense model that does not fit: the launch "
                    f"will fail.{suggestion}" + balanced_text,
                    dialog=True,
                )
            )
        else:
            out.append(
                Message(
                    base
                    + " The profile as configured may not fit."
                    + suggestion
                    + balanced_text,
                    dialog=True,
                )
            )
    if ram.fits is False:
        load_mode = (
            str(eff.get("load-mode", "auto") or "auto")
            if accepts(CATALOG["load-mode"], engine)
            else "auto"
        )
        locked = load_mode in ("none", "mlock", "mmap+mlock") or bool(eff.get("mlock"))
        tail = (
            "the launch will fail."
            if locked
            else "the server will page weights in and out of RAM and run slowly."
        )
        out.append(
            Message(
                f"RAM: est ~{_gib(ram.est)} GiB exceeds available "
                f"~{_gib(ram.available)} GiB by ~{_gib(-ram.margin)} GiB; "
                f"{tail}",
                dialog=True,
            )
        )
    return out


def _verdicts(est, free_bytes_per_gpu, ram_available):
    """(cards, ram) verdicts for one memory estimate against the launch
    node's free VRAM per card and available RAM."""
    cards = tuple(
        CardFit(i, c.total, int(f), int(f) - c.total, int(f) - c.total >= 0)
        for i, (c, f) in enumerate(zip(est.cards, free_bytes_per_gpu, strict=False))
    )
    if ram_available is None:
        ram = RamFit(est.ram.total, None, None, None)
    else:
        margin = int(ram_available) - est.ram.total
        ram = RamFit(est.ram.total, int(ram_available), margin, margin >= 0)
    return cards, ram


def fit_report(
    meta,
    weights_bytes,
    *,
    settings,
    engine,
    free_bytes_per_gpu,
    ram_available,
    raw_args="",
    draft_meta=None,
    draft_weights=0,
    mmproj_bytes=0,
    with_balanced=False,
):
    """Per-card and RAM verdicts plus messages for a profile; None when the
    estimate is unknowable or no card is visible. On ik_llama.cpp with
    --fit on, a MoE model with a tensor table that a card cannot hold is
    re-estimated with as many layers' experts kept in host RAM as it takes
    to fit, and the verdicts and fit outcome reflect that adjusted
    estimate. Without a tensor table the file-size fallback cannot move
    expert bytes, so the plain shortfall applies instead. The balanced
    split is computed, and carried in FitReport.balanced, only when
    with_balanced is true or a card is over budget: a readout that fits
    pays nothing for a search it will not show."""
    if not free_bytes_per_gpu:
        return None
    est = estimate_memory(
        meta,
        weights_bytes,
        settings=settings,
        engine=engine,
        free_bytes_per_gpu=free_bytes_per_gpu,
        raw_args=raw_args,
        draft_meta=draft_meta,
        draft_weights=draft_weights,
        mmproj_bytes=mmproj_bytes,
    )
    if est is None:
        return None
    cards, ram = _verdicts(est, free_bytes_per_gpu, ram_available)
    eff = _pl.effective_settings(settings, raw_args)
    ik_moe_layers = None
    if (
        engine == "ik_llama.cpp"
        and _fit_state(eff) == "on"
        and is_moe(meta)
        and meta.tensors
        and any(not c.fits for c in cards)
    ):
        adjusted, layers = _ik_moe_offload(
            meta,
            weights_bytes,
            settings=settings,
            engine=engine,
            free_bytes_per_gpu=free_bytes_per_gpu,
            raw_args=raw_args,
            draft_meta=draft_meta,
            draft_weights=draft_weights,
            mmproj_bytes=mmproj_bytes,
        )
        if adjusted is not None:
            est = adjusted
            cards, ram = _verdicts(est, free_bytes_per_gpu, ram_available)
            ik_moe_layers = layers
    balanced = None
    if with_balanced or any(not c.fits for c in cards):
        balanced = _balance.balanced_split(
            meta,
            weights_bytes,
            settings=settings,
            engine=engine,
            free_bytes_per_gpu=free_bytes_per_gpu,
            raw_args=raw_args,
            draft_meta=draft_meta,
            draft_weights=draft_weights,
            mmproj_bytes=mmproj_bytes,
        )
    messages = _messages(
        meta,
        weights_bytes,
        est,
        cards,
        ram,
        settings=settings,
        engine=engine,
        free_bytes_per_gpu=free_bytes_per_gpu,
        raw_args=raw_args,
        draft_meta=draft_meta,
        draft_weights=draft_weights,
        mmproj_bytes=mmproj_bytes,
        ik_moe_layers=ik_moe_layers,
        balanced=balanced,
    )
    return FitReport(
        est, cards, ram, tuple(messages), all(c.fits for c in cards), balanced
    )


def render_lines(report: FitReport) -> list:
    """One rich-text line per card and one for RAM; an over-budget line is
    wrapped in the red span the readout already uses."""
    est = report.estimate
    kv_label = "KV up to" if est.kv_upper_bound else "KV"
    lines = []
    for c, ce in zip(report.cards, est.cards, strict=False):
        parts = (
            f"weights {_gib(ce.weights)}, {kv_label} {_gib(ce.kv)}"
            f"{_state_parts(ce.state)}, "
            f"compute ~{_gib(ce.compute)}, overhead {_gib(ce.overhead)}"
        )
        if c.fits:
            lines.append(
                f"GPU{c.index}: est ~{_gib(c.est)} / ~{_gib(c.free)} GiB free "
                f"({parts}) margin {_gib(c.margin)} GiB \u2713"
            )
        else:
            lines.append(
                '<span style="color:#c62828">'
                f"GPU{c.index}: est ~{_gib(c.est)} GiB &gt; ~{_gib(c.free)} "
                f"GiB free ({parts}) short {_gib(-c.margin)} GiB</span>"
            )
    r = report.ram
    ram_parts = (
        f"weights {_gib(est.ram.weights)}, {kv_label} {_gib(est.ram.kv)}"
        f"{_state_parts(est.ram.state, est.ram.checkpoints)}, "
        f"buffers {_gib(est.ram.buffers)}"
    )
    if r.available is None:
        lines.append(f"RAM: est ~{_gib(r.est)} GiB ({ram_parts}), available unknown")
    elif r.fits:
        lines.append(
            f"RAM: est ~{_gib(r.est)} / ~{_gib(r.available)} GiB available "
            f"({ram_parts}) \u2713"
        )
    else:
        lines.append(
            '<span style="color:#c62828">'
            f"RAM: est ~{_gib(r.est)} GiB &gt; ~{_gib(r.available)} GiB "
            f"available ({ram_parts})</span>"
        )
    return lines


def plain_lines(report: FitReport) -> list[str]:
    """render_lines with the rich-text markup stripped: tags dropped,
    entities unescaped and the check mark spelled as ASCII."""
    out = []
    for line in render_lines(report):
        text = html.unescape(_TAG_RE.sub("", line))
        out.append(text.replace("\u2713", "ok"))
    return out


def render_tooltip(report: FitReport) -> str:
    est = report.estimate
    lines = [f"Context used for KV: {est.ctx}"]
    for i, c in enumerate(est.cards):
        lines.append(
            f"GPU{i}: weights {_gib(c.weights)} GiB, KV {_gib(c.kv)} GiB"
            f"{_state_parts(c.state, unit='GiB')}, "
            f"compute {_gib(c.compute)} GiB, overhead {_gib(c.overhead)} GiB"
        )
    lines.append(
        f"RAM: weights {_gib(est.ram.weights)} GiB, KV {_gib(est.ram.kv)} GiB"
        f"{_state_parts(est.ram.state, est.ram.checkpoints, 'GiB')}, "
        f"host buffers {_gib(est.ram.buffers)} GiB"
    )
    lines.append(
        "compute is a formula estimate scaled by --ubatch-size; overhead is "
        "the backend context per card"
    )
    if est.kv_upper_bound:
        lines.append(
            "KV is an upper bound: a sliding-window model keeps less at full context"
        )
    for m in report.messages:
        lines.append(m.text)
    return "\n".join(lines)


def render_dialog(report: FitReport) -> str | None:
    """The launch preflight dialog text: the same breakdown the readout
    shows, plain text, followed by a blank line and the messages marked
    for the dialog. None when nothing needs the user's eye."""
    texts = [m.text for m in report.messages if m.dialog]
    if not texts:
        return None
    return "\n".join(plain_lines(report)) + "\n\n" + "\n\n".join(texts)


def to_json(report: FitReport) -> dict:
    est = report.estimate
    out = {
        "fits": report.fits,
        "ctx": est.ctx,
        "kv_upper_bound": est.kv_upper_bound,
        "cards": [
            {
                "index": c.index,
                "est": c.est,
                "free": c.free,
                "margin": c.margin,
                "fits": c.fits,
                "weights": ce.weights,
                "kv": ce.kv,
                "compute": ce.compute,
                "overhead": ce.overhead,
                "state": ce.state,
            }
            for c, ce in zip(report.cards, est.cards, strict=False)
        ],
        "ram": {
            "est": report.ram.est,
            "available": report.ram.available,
            "margin": report.ram.margin,
            "fits": report.ram.fits,
            "weights": est.ram.weights,
            "kv": est.ram.kv,
            "buffers": est.ram.buffers,
            "state": est.ram.state,
            "checkpoints": est.ram.checkpoints,
        },
        "messages": [m.text for m in report.messages],
    }
    if report.balanced is not None:
        out["balanced_split"] = {
            "value": report.balanced.value,
            "layers_per_card": list(report.balanced.layers_per_card),
            "boundary_layers": list(report.balanced.boundary_layers),
            "fits": report.balanced.fits,
        }
    return out
