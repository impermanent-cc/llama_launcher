import struct
from llama_launcher.core.gguf import parse_gguf_header, GgufMeta


def _kv_str(key, val):
    kb = key.encode()
    vb = val.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb


def _kv_u32(key, val):
    kb = key.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 4) + struct.pack("<I", val)


def _kv_arr_u32(key, vals):
    kb = key.encode()
    out = struct.pack("<Q", len(kb)) + kb
    out += struct.pack("<I", 9)             # value type = _ARR
    out += struct.pack("<I", 4)             # element type = _U32
    out += struct.pack("<Q", len(vals))     # count
    return out + b"".join(struct.pack("<I", v) for v in vals)


def _arr_head_gguf():
    # A model that stores per-layer attention.head_count[_kv] as arrays
    # (its layers differ); valid GGUF that some real models emit.
    kvs = [
        _kv_str("general.architecture", "hybrid"),
        _kv_u32("hybrid.block_count", 4),
        _kv_arr_u32("hybrid.attention.head_count", [32, 32, 32, 32]),
        _kv_arr_u32("hybrid.attention.head_count_kv", [8, 8, 4, 4]),
        _kv_u32("hybrid.embedding_length", 4096),
        _kv_u32("hybrid.context_length", 8192),
    ]
    return (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
            + struct.pack("<Q", len(kvs)) + b"".join(kvs))


def _synthetic_gguf():
    kvs = [
        _kv_str("general.architecture", "qwen3"),
        _kv_str("general.name", "Test Qwen"),
        _kv_str("general.size_label", "30B-A3B"),
        _kv_u32("general.file_type", 26),          # IQ3_S
        _kv_u32("qwen3.block_count", 48),
        _kv_u32("qwen3.attention.head_count", 32),
        _kv_u32("qwen3.attention.head_count_kv", 4),
        _kv_u32("qwen3.embedding_length", 4096),
        _kv_u32("qwen3.context_length", 40960),
    ]
    body = b"".join(kvs)
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs))
    return header + body


def test_parse_basic():
    m = parse_gguf_header(_synthetic_gguf())
    assert m.arch == "qwen3"
    assert m.name == "Test Qwen"
    assert m.size_label == "30B-A3B"
    assert m.quant == "IQ3_S"
    assert m.n_layers == 48
    assert m.n_head == 32
    assert m.n_head_kv == 4
    assert m.n_embd == 4096
    assert m.ctx_train == 40960


def test_head_kv_falls_back_to_head():
    import struct as s
    kvs = [_kv_str("general.architecture", "llama"), _kv_u32("llama.attention.head_count", 16)]
    blob = b"GGUF" + s.pack("<I", 3) + s.pack("<Q", 0) + s.pack("<Q", len(kvs)) + b"".join(kvs)
    m = parse_gguf_header(blob)
    assert m.n_head == 16 and m.n_head_kv == 16


def test_bad_magic_raises():
    import pytest
    with pytest.raises(ValueError):
        parse_gguf_header(b"NOPE" + b"\x00" * 20)


def test_truncated_raises():
    import pytest
    with pytest.raises(ValueError):
        parse_gguf_header(_synthetic_gguf()[:12])


def test_parses_capability_kv():
    kvs = [_kv_str("general.architecture", "qwen35moe"),
           _kv_u32("qwen35moe.expert_count", 256),
           _kv_u32("qwen35moe.attention.sliding_window", 1024),
           _kv_u32("qwen35moe.nextn_predict_layers", 1)]
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs)) + b"".join(kvs)
    m = parse_gguf_header(blob)
    assert m.expert_count == 256
    assert m.sliding_window == 1024
    assert m.nextn_predict_layers == 1


def test_capability_kv_absent_is_none():
    m = parse_gguf_header(_synthetic_gguf())   # has none of these keys
    assert m.expert_count is None
    assert m.sliding_window is None
    assert m.nextn_predict_layers is None


def test_array_head_counts_collapse_to_scalar():
    # GgufMeta is scalar (int | None); per-layer array head counts must
    # collapse to a single int, the max, a conservative scalar for the
    # VRAM estimate (and exact when every layer is equal).
    m = parse_gguf_header(_arr_head_gguf())
    assert m.n_head == 32 and isinstance(m.n_head, int)
    assert m.n_head_kv == 8 and isinstance(m.n_head_kv, int)
    assert m.n_layers == 4 and m.n_embd == 4096 and m.ctx_train == 8192


def test_array_head_counts_feed_vram_estimate_without_crash():
    # The collapsed scalar head counts feed vram.estimate() without raising.
    from llama_launcher.core import vram
    m = parse_gguf_header(_arr_head_gguf())
    est = vram.estimate(n_layers=m.n_layers, n_head=m.n_head or 1,
                        n_head_kv=m.n_head_kv or m.n_head or 1,
                        n_embd=m.n_embd, ctx=m.ctx_train)
    assert est.total_bytes > 0


def test_parses_pooling_type():
    kvs = [_kv_str("general.architecture", "bert"),
           _kv_u32("bert.pooling_type", 2)]
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs)) + b"".join(kvs)
    m = parse_gguf_header(blob)
    assert m.pooling_type == 2


def test_pooling_type_absent_is_none():
    m = parse_gguf_header(_synthetic_gguf())
    assert m.pooling_type is None
