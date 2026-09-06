import struct
from dataclasses import dataclass

GGUF_MAGIC = b"GGUF"

(_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STR, _ARR, _U64, _I64, _F64) = range(
    13
)

_FIXED = {
    _U8: ("<B", 1),
    _I8: ("<b", 1),
    _U16: ("<H", 2),
    _I16: ("<h", 2),
    _U32: ("<I", 4),
    _I32: ("<i", 4),
    _F32: ("<f", 4),
    _BOOL: ("<?", 1),
    _U64: ("<Q", 8),
    _I64: ("<q", 8),
    _F64: ("<d", 8),
}

_FTYPE = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
    38: "MXFP4_MOE",
    39: "NVFP4",
    40: "Q1_0",
}


def ftype_name(ftype) -> str:
    if ftype is None:
        return ""
    return _FTYPE.get(ftype, f"ftype{ftype}")


def _num(v):
    """Reduce a GGUF numeric metadata value to a single int.

    Some per-layer fields (notably attention.head_count[_kv]) are stored as a
    one-entry-per-block array when a model's layers differ; GgufMeta is scalar
    (int | None), so collapse arrays to their max, a conservative scalar for
    the VRAM estimate, and exact when every layer is equal. Pass scalars and
    None through unchanged; an empty/non-numeric array yields None.
    """
    if isinstance(v, (list, tuple)):
        nums = [x for x in v if isinstance(x, (int, float))]
        return max(nums) if nums else None
    return v


def _per_layer(v):
    """A GGUF value that names one number per layer as a tuple of ints, or
    None when it is a scalar, empty or non-numeric. It keeps the per-layer
    detail _num collapses, for the fields whose layers really differ."""
    if not isinstance(v, (list, tuple)) or not v:
        return None
    nums = [x for x in v if isinstance(x, (int, float))]
    return tuple(int(x) for x in nums) if nums else None


@dataclass(frozen=True)
class TensorInfo:
    name: str
    n_elements: int
    ggml_type: int
    nbytes: int


# (elements per block, bytes per block) per ggml_type id, from ggml-common.h.
GGML_TYPE_SIZES = {
    0: (1, 4),  # F32
    1: (1, 2),  # F16
    2: (32, 18),  # Q4_0
    3: (32, 20),  # Q4_1
    6: (32, 22),  # Q5_0
    7: (32, 24),  # Q5_1
    8: (32, 34),  # Q8_0
    9: (32, 36),  # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),  # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),  # I8
    25: (1, 2),  # I16
    26: (1, 4),  # I32
    27: (1, 8),  # I64
    28: (1, 8),  # F64
    29: (256, 56),  # IQ1_M
    30: (1, 2),  # BF16
    34: (256, 54),  # TQ1_0
    35: (256, 66),  # TQ2_0
    39: (32, 17),  # MXFP4
    40: (64, 36),  # NVFP4
    41: (128, 18),  # Q1_0
    42: (64, 18),  # Q2_0
}


def tensor_nbytes(n_elements: int, ggml_type: int) -> int:
    """Bytes a tensor occupies in a buffer. A partial trailing block counts
    as a whole block; an unknown type counts two bytes per element."""
    block, size = GGML_TYPE_SIZES.get(ggml_type, (1, 2))
    return -(-int(n_elements) // block) * size


@dataclass
class GgufMeta:
    arch: str = ""
    name: str = ""
    n_layers: int | None = None
    n_head: int | None = None
    n_head_kv: int | None = None
    n_embd: int | None = None
    ctx_train: int | None = None
    size_label: str = ""
    quant: str = ""
    expert_count: int | None = None
    sliding_window: int | None = None
    nextn_predict_layers: int | None = None
    pooling_type: int | None = None
    n_ff: int | None = None
    n_ff_exp: int | None = None
    n_expert_used: int | None = None
    n_vocab: int | None = None
    split_count: int = 1
    tensors: tuple = ()
    full_attention_interval: int | None = None
    kv_layer_heads: tuple | None = None
    head_dim_k: int | None = None
    head_dim_v: int | None = None
    ssm_conv_kernel: int | None = None
    ssm_inner_size: int | None = None
    ssm_state_size: int | None = None
    ssm_group_count: int | None = None
    ff_layers: tuple | None = None


class _Reader:
    def __init__(self, data: bytes):
        self.d = data
        self.off = 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.off + n > len(self.d):
            raise ValueError("gguf: unexpected end of data")
        b = self.d[self.off : self.off + n]
        self.off += n
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def gstr(self) -> str:
        return self.take(self.u64()).decode("utf-8", "replace")

    def value(self, vtype: int):
        if vtype in _FIXED:
            fmt, size = _FIXED[vtype]
            return struct.unpack(fmt, self.take(size))[0]
        if vtype == _STR:
            return self.gstr()
        if vtype == _ARR:
            elem = self.u32()
            count = self.u64()
            return [self.value(elem) for _ in range(count)]
        raise ValueError(f"gguf: unknown value type {vtype}")


def _parse_tensor_table(r: "_Reader", count: int) -> tuple:
    """The tensor infos that follow the key-value block: name, dimensions,
    type and data offset. A table the read limit cuts short yields an empty
    tuple, since the key-value block before it is complete on its own."""
    if count > 100_000:
        return ()
    out = []
    try:
        for _ in range(count):
            name = r.gstr()
            n_dims = r.u32()
            if n_dims > 8:
                return ()
            n_elements = 1
            for _ in range(n_dims):
                n_elements *= r.u64()
            ggml_type = r.u32()
            r.u64()  # byte offset inside the data section
            out.append(
                TensorInfo(
                    name, n_elements, ggml_type, tensor_nbytes(n_elements, ggml_type)
                )
            )
    except ValueError:
        return ()
    return tuple(out)


def parse_gguf_header(data: bytes) -> GgufMeta:
    r = _Reader(data)
    if r.take(4) != GGUF_MAGIC:
        raise ValueError("gguf: bad magic")
    version = r.u32()
    if version not in (2, 3):
        raise ValueError(f"gguf: unsupported version {version}")
    tensor_count = r.u64()
    kv_count = r.u64()
    if kv_count > 1_000_000:
        raise ValueError("gguf: implausible kv_count")
    kv: dict = {}
    for _ in range(kv_count):
        key = r.gstr()
        kv[key] = r.value(r.u32())

    arch = kv.get("general.architecture", "") or ""

    def a(suffix):
        return kv.get(f"{arch}.{suffix}")

    n_head = _num(a("attention.head_count"))
    raw_kv_heads = a("attention.head_count_kv")
    n_head_kv = _num(raw_kv_heads)
    if n_head_kv is None:
        n_head_kv = n_head

    kv_layer_heads = _per_layer(raw_kv_heads)
    ff_layers = _per_layer(a("feed_forward_length"))

    tensors = _parse_tensor_table(r, tensor_count)
    tokens = kv.get("tokenizer.ggml.tokens")
    n_vocab = (
        len(tokens) if isinstance(tokens, list) and tokens else _num(a("vocab_size"))
    )
    split_count = kv.get("split.count")

    return GgufMeta(
        arch=arch,
        name=kv.get("general.name", "") or "",
        n_layers=_num(a("block_count")),
        n_head=n_head,
        n_head_kv=n_head_kv,
        n_embd=_num(a("embedding_length")),
        ctx_train=_num(a("context_length")),
        size_label=kv.get("general.size_label", "") or "",
        quant=ftype_name(kv.get("general.file_type")),
        expert_count=_num(a("expert_count")),
        sliding_window=_num(a("attention.sliding_window")),
        nextn_predict_layers=_num(a("nextn_predict_layers")),
        pooling_type=_num(a("pooling_type")),
        n_ff=_num(a("feed_forward_length")),
        n_ff_exp=_num(a("expert_feed_forward_length")),
        n_expert_used=_num(a("expert_used_count")),
        n_vocab=n_vocab,
        split_count=int(split_count)
        if isinstance(split_count, int) and split_count > 0
        else 1,
        tensors=tensors,
        full_attention_interval=_num(a("full_attention_interval")),
        kv_layer_heads=kv_layer_heads,
        head_dim_k=_num(a("attention.key_length")),
        head_dim_v=_num(a("attention.value_length")),
        ssm_conv_kernel=_num(a("ssm.conv_kernel")),
        ssm_inner_size=_num(a("ssm.inner_size")),
        ssm_state_size=_num(a("ssm.state_size")),
        ssm_group_count=_num(a("ssm.group_count")),
        ff_layers=ff_layers,
    )
