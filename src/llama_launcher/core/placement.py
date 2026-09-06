"""Where each tensor of a model lands: on a card or in host RAM, under the
same rules llama.cpp and ik_llama.cpp apply to --n-gpu-layers, the CPU
offload flags, --override-tensor and the layer split across cards.
ALL_EXPS_PATTERN is upstream fit's all-experts pattern, pinned here by a
fixture so a change to it upstream is caught by the test."""

import re
from dataclasses import dataclass

from .command_builder import ACCUMULATING_KEYS, raw_arg_values
from .settings_catalog import CATALOG, accepts

EXPS_REGEX = r"\.ffn_(up|down|gate|gate_up)_(ch|)exps"
DENSE_FFN_REGEX = r"\.ffn_(up|down|gate)\."
ALL_EXPS_PATTERN = r"blk\.\d+\.ffn_(up|down|gate_up|gate)_(ch|)exps"

_BLK_RE = re.compile(r"^blk\.(\d+)\.")


@dataclass(frozen=True)
class CpuRule:
    """Tensors matching `regex` in block layers below `max_layer` (every
    layer when None) stay in host RAM."""

    max_layer: int | None
    regex: str


@dataclass(frozen=True)
class Placement:
    n_layers: int
    layer_gpu: tuple
    layer_cpu: tuple
    output_gpu: int
    output_cpu: int
    input_cpu: int
    devices: tuple  # per layer index 0..n_layers, True on a card; index n_layers is the output layer

    @property
    def gpu_bytes(self) -> int:
        return sum(self.layer_gpu) + self.output_gpu

    @property
    def cpu_bytes(self) -> int:
        return sum(self.layer_cpu) + self.output_cpu + self.input_cpu


@dataclass(frozen=True)
class Distribution:
    # per layer index 0..n_layers: fraction of its GPU bytes per card
    weight_share: tuple
    # per block layer: card index holding its KV cache, None in RAM
    kv_card: tuple


def effective_settings(settings: dict, raw_args: str) -> dict:
    """Profile settings with raw-arg spellings of catalogued flags laid over
    them, since the later argv token wins on the server. The accumulating
    keys (--override-tensor, --spec-draft-override-tensor) instead join the
    raw value after the profile's own, matching how argv carries both
    rather than the raw value replacing it; a blank or missing profile
    value leaves the raw value alone, and a trailing comma on the profile
    value is stripped first so no empty entry appears."""
    out = dict(settings)
    raw = raw_arg_values(raw_args or "")
    for key in ACCUMULATING_KEYS:
        raw_value = raw.get(key)
        if not isinstance(raw_value, str):
            continue
        prior = str(out.get(key, "") or "").strip().rstrip(",").strip()
        if prior:
            raw[key] = f"{prior},{raw_value}"
    out.update(raw)
    return out


def int_or_none(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def gpu_layer_count(value, n_layers: int, engine: str) -> int:
    """The layer count -ngl resolves to: 0 to n_layers + 1, where the extra
    one is the output layer. 'all', a negative number and an unreadable
    value mean every layer; 'auto' means every layer on llama.cpp and no
    layer on ik_llama.cpp, whose launch receives no flag for it."""
    full = int(n_layers) + 1
    if value == "auto":
        return 0 if engine == "ik_llama.cpp" else full
    if value == "all":
        return full
    n = int_or_none(value)
    if n is None or n < 0:
        return 0 if engine == "ik_llama.cpp" else full
    return min(n, full)


def layer_devices(n_layers: int, n_gpu_layers: int, engine: str) -> tuple:
    """True for each layer index that lives on a card. Index n_layers is
    the output layer. llama.cpp counts the output layer first and then
    block layers from the last one down; ik_llama.cpp counts block layers
    from the last one down and adds the output layer only past them."""
    n_layers = int(n_layers)
    ngl = max(0, min(int(n_gpu_layers), n_layers + 1))
    if engine == "ik_llama.cpp":
        blocks = min(ngl, n_layers)
        start = n_layers - blocks
        return tuple([il >= start for il in range(n_layers)] + [ngl > n_layers])
    start = n_layers + 1 - ngl
    return tuple(il >= start for il in range(n_layers + 1))


def _count(settings: dict, engine: str, key: str) -> int:
    setting = CATALOG.get(key)
    if setting is None or not accepts(setting, engine):
        return 0
    n = int_or_none(settings.get(key))
    return n if n and n > 0 else 0


def _flag(settings: dict, engine: str, key: str) -> bool:
    setting = CATALOG.get(key)
    if setting is None or not accepts(setting, engine):
        return False
    return bool(settings.get(key))


def cpu_rules(settings: dict, engine: str, *, draft: bool = False) -> list:
    """The CPU-offload flags of the main model, or of the draft model with
    the draft twins, as rules; a flag the engine does not accept adds no
    rule. --n-cpu-ffn has no draft twin."""
    pfx = "spec-draft-" if draft else ""
    rules = []
    if _flag(settings, engine, pfx + "cpu-moe"):
        rules.append(CpuRule(None, EXPS_REGEX))
    n = _count(settings, engine, pfx + "n-cpu-moe")
    if n:
        rules.append(CpuRule(n, EXPS_REGEX))
    if not draft:
        n = _count(settings, engine, "n-cpu-ffn")
        if n:
            rules.append(CpuRule(n, DENSE_FFN_REGEX))
    return rules


def parse_overrides(value) -> list:
    """--override-tensor entries as (pattern, to_cpu) in order; an entry
    without '=' or with a pattern the regex engine rejects is dropped."""
    out = []
    for entry in str(value or "").split(","):
        entry = entry.strip()
        if "=" not in entry:
            continue
        pattern, _, target = entry.partition("=")
        pattern = pattern.strip()
        try:
            re.compile(pattern)
        except re.error:
            continue
        out.append((pattern, target.strip().upper() == "CPU"))
    return out


def _layer_of(name: str, n_layers: int):
    """Block index, n_layers for the output layer, None for input tensors."""
    m = _BLK_RE.match(name)
    if m:
        il = int(m.group(1))
        return il if il < n_layers else None
    if name.startswith("output"):
        return n_layers
    return None


def place(
    tensors, n_layers, *, devices, rules, overrides, weights_fallback=0
) -> Placement:
    """Bytes per layer on a card and in RAM. A tensor starts on whichever
    side --n-gpu-layers put its layer on; on a card, a CPU rule can still
    move it to RAM, and that claim is final. Every tensor a CPU rule did
    not claim, whether it started on a card or in RAM, is then checked
    against --override-tensor in order, and the first match decides: a
    CPU target means RAM, any other target means a card. Input tensors
    (token embeddings and any other tensor outside a block) stay in RAM.
    When the table has no output.weight, the model ties its output matrix
    to token_embd.weight, so those bytes also count at the output layer's
    device, on top of their input-layer count in RAM. An override pattern
    the regex engine rejects is dropped rather than raised. With no tensor
    table, `weights_fallback` bytes stand for the whole model and go to a
    card whenever any layer index, block or output, is on one, since that
    means some offload happened; otherwise they go to RAM."""
    n_layers = int(n_layers)
    layer_gpu = [0] * n_layers
    layer_cpu = [0] * n_layers
    output_gpu = output_cpu = input_cpu = 0
    if not tensors:
        if any(devices):
            output_gpu = int(weights_fallback)
        else:
            output_cpu = int(weights_fallback)
        return Placement(
            n_layers,
            tuple(layer_gpu),
            tuple(layer_cpu),
            output_gpu,
            output_cpu,
            input_cpu,
            tuple(devices),
        )
    compiled_rules = [(r.max_layer, re.compile(r.regex)) for r in rules]
    compiled_overrides = []
    for p, to_cpu in overrides:
        try:
            compiled_overrides.append((re.compile(p), to_cpu))
        except re.error:
            continue
    tensor_names = {t.name for t in tensors}
    tied_embeddings = "output.weight" not in tensor_names
    for t in tensors:
        il = _layer_of(t.name, n_layers)
        if il is None:
            input_cpu += t.nbytes
            if tied_embeddings and t.name == "token_embd.weight":
                if devices[n_layers]:
                    output_gpu += t.nbytes
                else:
                    output_cpu += t.nbytes
            continue
        on_gpu = bool(devices[il])
        cpu_claimed = False
        if on_gpu:
            for max_layer, rx in compiled_rules:
                if (max_layer is None or il < max_layer) and rx.search(t.name):
                    on_gpu = False
                    cpu_claimed = True
                    break
        if not cpu_claimed:
            for rx, to_cpu in compiled_overrides:
                if rx.search(t.name):
                    on_gpu = not to_cpu
                    break
        if il == n_layers:
            if on_gpu:
                output_gpu += t.nbytes
            else:
                output_cpu += t.nbytes
        elif on_gpu:
            layer_gpu[il] += t.nbytes
        else:
            layer_cpu[il] += t.nbytes
    return Placement(
        n_layers,
        tuple(layer_gpu),
        tuple(layer_cpu),
        output_gpu,
        output_cpu,
        input_cpu,
        tuple(devices),
    )


def split_fractions(tensor_split, free_per_card, n_cards: int) -> list:
    """Cumulative split points per card, the last one 1.0: from
    --tensor-split (comma or slash separated, missing tail entries zero),
    else each card's free memory, else equal shares."""
    n_cards = max(1, int(n_cards))
    vals = []
    for tok in re.split(r"[,/]+", str(tensor_split or "")):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(max(0.0, float(tok)))
        except ValueError:
            vals = []
            break
    vals = (vals + [0.0] * n_cards)[:n_cards]
    if sum(vals) <= 0:
        vals = [
            max(0.0, float(b)) if b is not None else 0.0
            for b in list(free_per_card)[:n_cards]
        ]
        vals = (vals + [0.0] * n_cards)[:n_cards]
    if sum(vals) <= 0:
        vals = [1.0] * n_cards
    total = sum(vals)
    out = []
    acc = 0.0
    for v in vals:
        acc += v
        out.append(acc / total)
    out[-1] = 1.0
    return out


def distribute(
    placement: Placement, *, free_per_card, tensor_split, split_mode, main_gpu, engine
) -> Distribution:
    """Which card holds each GPU layer's weights, and which card holds
    each GPU layer's KV cache. A layer's weights get a card share when
    --n-gpu-layers already puts it on a card, or when an override promoted
    some of its bytes onto one although --n-gpu-layers keeps it in RAM;
    the KV cache follows --n-gpu-layers alone, since --override-tensor
    only ever names weight tensors. Layer (and ik's graph) mode gives each
    card a contiguous run of layers by the split points, the way the
    engines do with (il - first GPU layer) / GPU layer count against the
    cumulative split; a promoted layer uses the same formula against its
    own index, clamped to the first card when that falls below zero. None
    mode puts everything on --main-gpu; row mode spreads every GPU layer's
    weights by the split and keeps every KV cache on --main-gpu."""
    n_cards = max(1, len(free_per_card))
    n_layers = placement.n_layers
    devices = placement.devices
    main = int_or_none(main_gpu) or 0
    if not 0 <= main < n_cards:
        main = 0
    zero = tuple(0.0 for _ in range(n_cards))

    def one(card):
        return tuple(1.0 if i == card else 0.0 for i in range(n_cards))

    def on_card(il):
        if devices[il]:
            return True
        if il < n_layers:
            return placement.layer_gpu[il] > 0
        return placement.output_gpu > 0

    if split_mode == "none":
        share = [one(main) if on_card(il) else zero for il in range(n_layers + 1)]
        kv = [main if devices[il] else None for il in range(n_layers)]
        return Distribution(tuple(share), tuple(kv))
    points = split_fractions(tensor_split, free_per_card, n_cards)
    if split_mode == "row":
        fractions = [points[0]] + [points[i] - points[i - 1] for i in range(1, n_cards)]
        share = [
            tuple(fractions) if on_card(il) else zero for il in range(n_layers + 1)
        ]
        kv = [main if devices[il] else None for il in range(n_layers)]
        return Distribution(tuple(share), tuple(kv))
    gpu_indices = [il for il in range(n_layers + 1) if devices[il]]
    act = len(gpu_indices)
    start = gpu_indices[0] if gpu_indices else n_layers + 1
    share = []
    kv = []
    for il in range(n_layers + 1):
        if not on_card(il):
            share.append(zero)
            if il < n_layers:
                kv.append(None)
            continue
        frac = (il - start) / act if act else -1.0
        if frac < 0:
            card = 0
        else:
            card = next((i for i, p in enumerate(points) if p > frac), n_cards - 1)
        share.append(one(card))
        if il < n_layers:
            kv.append(card if devices[il] else None)
    return Distribution(tuple(share), tuple(kv))
