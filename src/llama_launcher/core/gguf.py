import struct
from dataclasses import dataclass

GGUF_MAGIC = b"GGUF"

(_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STR, _ARR, _U64, _I64, _F64) = range(13)

_FIXED = {
    _U8: ("<B", 1), _I8: ("<b", 1), _U16: ("<H", 2), _I16: ("<h", 2),
    _U32: ("<I", 4), _I32: ("<i", 4), _F32: ("<f", 4), _BOOL: ("<?", 1),
    _U64: ("<Q", 8), _I64: ("<q", 8), _F64: ("<d", 8),
}

_FTYPE = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
    38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0",
}


def ftype_name(ftype) -> str:
    if ftype is None:
        return ""
    return _FTYPE.get(ftype, f"ftype{ftype}")


def _num(v):
    """Reduce a GGUF numeric metadata value to a single int.

    Some per-layer fields (notably attention.head_count[_kv]) are stored as a
    one-entry-per-block array when a model's layers differ; GgufMeta is scalar
    (int | None), so collapse arrays to their max — a conservative scalar for
    the VRAM estimate, and exact when every layer is equal. Pass scalars and
    None through unchanged; an empty/non-numeric array yields None.
    """
    if isinstance(v, (list, tuple)):
        nums = [x for x in v if isinstance(x, (int, float))]
        return max(nums) if nums else None
    return v


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


class _Reader:
    def __init__(self, data: bytes):
        self.d = data
        self.off = 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.off + n > len(self.d):
            raise ValueError("gguf: unexpected end of data")
        b = self.d[self.off:self.off + n]
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


def parse_gguf_header(data: bytes) -> GgufMeta:
    r = _Reader(data)
    if r.take(4) != GGUF_MAGIC:
        raise ValueError("gguf: bad magic")
    version = r.u32()
    if version not in (2, 3):
        raise ValueError(f"gguf: unsupported version {version}")
    r.u64()                      # tensor_count (unused)
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
    n_head_kv = _num(a("attention.head_count_kv"))
    if n_head_kv is None:
        n_head_kv = n_head

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
    )
