import struct

from llama_launcher.core.gguf import parse_gguf_header, tensor_nbytes


def _kv_str(key, val):
    kb = key.encode()
    vb = val.encode()
    return (
        struct.pack("<Q", len(kb))
        + kb
        + struct.pack("<I", 8)
        + struct.pack("<Q", len(vb))
        + vb
    )


def _kv_u32(key, val):
    kb = key.encode()
    return (
        struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 4) + struct.pack("<I", val)
    )


def _kv_arr_u32(key, vals):
    kb = key.encode()
    out = struct.pack("<Q", len(kb)) + kb
    out += struct.pack("<I", 9)  # value type = _ARR
    out += struct.pack("<I", 4)  # element type = _U32
    out += struct.pack("<Q", len(vals))  # count
    return out + b"".join(struct.pack("<I", v) for v in vals)


def _kv_arr_arr_u32(key, vals):
    # An array of arrays: value type 9 (_ARR) nested inside value type 9,
    # each inner element itself an array of u32.
    kb = key.encode()
    out = struct.pack("<Q", len(kb)) + kb
    out += struct.pack("<I", 9)  # value type = _ARR
    out += struct.pack("<I", 9)  # element type = _ARR
    out += struct.pack("<Q", len(vals))  # outer count
    for inner in vals:
        out += struct.pack("<I", 4)  # inner element type = _U32
        out += struct.pack("<Q", len(inner))  # inner count
        out += b"".join(struct.pack("<I", v) for v in inner)
    return out


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
    return (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", len(kvs))
        + b"".join(kvs)
    )


def _synthetic_gguf():
    kvs = [
        _kv_str("general.architecture", "qwen3"),
        _kv_str("general.name", "Test Qwen"),
        _kv_str("general.size_label", "30B-A3B"),
        _kv_u32("general.file_type", 26),  # IQ3_S
        _kv_u32("qwen3.block_count", 48),
        _kv_u32("qwen3.attention.head_count", 32),
        _kv_u32("qwen3.attention.head_count_kv", 4),
        _kv_u32("qwen3.embedding_length", 4096),
        _kv_u32("qwen3.context_length", 40960),
    ]
    body = b"".join(kvs)
    header = (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", len(kvs))
    )
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

    kvs = [
        _kv_str("general.architecture", "llama"),
        _kv_u32("llama.attention.head_count", 16),
    ]
    blob = (
        b"GGUF"
        + s.pack("<I", 3)
        + s.pack("<Q", 0)
        + s.pack("<Q", len(kvs))
        + b"".join(kvs)
    )
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
    kvs = [
        _kv_str("general.architecture", "qwen35moe"),
        _kv_u32("qwen35moe.expert_count", 256),
        _kv_u32("qwen35moe.attention.sliding_window", 1024),
        _kv_u32("qwen35moe.nextn_predict_layers", 1),
    ]
    blob = (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", len(kvs))
        + b"".join(kvs)
    )
    m = parse_gguf_header(blob)
    assert m.expert_count == 256
    assert m.sliding_window == 1024
    assert m.nextn_predict_layers == 1


def test_capability_kv_absent_is_none():
    m = parse_gguf_header(_synthetic_gguf())  # has none of these keys
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


def test_array_head_counts_feed_kv_cache_bytes_without_crash():
    # The collapsed scalar head counts feed the KV-cache byte math without
    # raising.
    from llama_launcher.core import vram

    m = parse_gguf_header(_arr_head_gguf())
    head_dim = m.n_embd // (m.n_head or 1)
    kv = vram.kv_cache_bytes(
        m.n_layers, m.n_head_kv or m.n_head or 1, head_dim, m.ctx_train
    )
    assert kv > 0


def test_parses_pooling_type():
    kvs = [_kv_str("general.architecture", "bert"), _kv_u32("bert.pooling_type", 2)]
    blob = (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", len(kvs))
        + b"".join(kvs)
    )
    m = parse_gguf_header(blob)
    assert m.pooling_type == 2


def test_pooling_type_absent_is_none():
    m = parse_gguf_header(_synthetic_gguf())
    assert m.pooling_type is None


def _tensor(name, dims, ggml_type):
    nb = name.encode()
    out = struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims))
    out += b"".join(struct.pack("<Q", d) for d in dims)
    return out + struct.pack("<I", ggml_type) + struct.pack("<Q", 0)


def _blob(kvs, tensors=()):
    return (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", len(tensors))
        + struct.pack("<Q", len(kvs))
        + b"".join(kvs)
        + b"".join(tensors)
    )


def _kv_u16(key, val):
    kb = key.encode()
    return (
        struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 2) + struct.pack("<H", val)
    )


def _kv_arr_str(key, vals):
    kb = key.encode()
    out = struct.pack("<Q", len(kb)) + kb
    out += struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(vals))
    for v in vals:
        vb = v.encode()
        out += struct.pack("<Q", len(vb)) + vb
    return out


def test_tensor_nbytes_by_type():
    assert tensor_nbytes(1024, 0) == 4096  # f32
    assert tensor_nbytes(1024, 1) == 2048  # f16
    assert tensor_nbytes(1024, 2) == 32 * 18  # q4_0: 32 elements per 18 bytes
    assert tensor_nbytes(1024, 12) == 4 * 144  # q4_K: 256 elements per 144 bytes
    assert tensor_nbytes(1000, 2) == 32 * 18  # partial block rounds up
    assert tensor_nbytes(1024, 99) == 2048  # unknown type counts two bytes each


def test_parse_tensor_table():
    kvs = [
        _kv_str("general.architecture", "llama"),
        _kv_u32("llama.block_count", 2),
        _kv_u32("llama.feed_forward_length", 512),
        _kv_u32("llama.vocab_size", 1000),
    ]
    tensors = [
        _tensor("token_embd.weight", [64, 1000], 1),
        _tensor("blk.0.attn_q.weight", [64, 64], 2),
        _tensor("blk.0.ffn_up.weight", [64, 512], 12),
    ]
    m = parse_gguf_header(_blob(kvs, tensors))
    assert [t.name for t in m.tensors] == [
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.0.ffn_up.weight",
    ]
    assert m.tensors[0].n_elements == 64000
    assert m.tensors[0].nbytes == 128000
    assert m.tensors[1].ggml_type == 2
    assert m.tensors[2].nbytes == (64 * 512 // 256) * 144
    assert m.n_ff == 512
    assert m.n_vocab == 1000
    assert m.split_count == 1


def test_truncated_tensor_table_yields_kv_only():
    kvs = [_kv_str("general.architecture", "llama"), _kv_u32("llama.block_count", 2)]
    tensors = [_tensor("blk.0.attn_q.weight", [64, 64], 2)] * 3
    blob = _blob(kvs, tensors)
    m = parse_gguf_header(blob[:-20])
    assert m.n_layers == 2
    assert m.tensors == ()


def test_vocab_from_token_array_beats_vocab_size_key():
    kvs = [
        _kv_str("general.architecture", "llama"),
        _kv_u32("llama.vocab_size", 5),
        _kv_arr_str("tokenizer.ggml.tokens", ["a", "b", "c"]),
    ]
    assert parse_gguf_header(_blob(kvs)).n_vocab == 3


def test_moe_hparams_and_split_count():
    kvs = [
        _kv_str("general.architecture", "qwen3moe"),
        _kv_u32("qwen3moe.expert_count", 128),
        _kv_u32("qwen3moe.expert_used_count", 8),
        _kv_u32("qwen3moe.expert_feed_forward_length", 768),
        _kv_u32("qwen3moe.feed_forward_length", 6144),
        _kv_u16("split.count", 3),
    ]
    m = parse_gguf_header(_blob(kvs))
    assert m.n_expert_used == 8
    assert m.n_ff_exp == 768
    assert m.n_ff == 6144
    assert m.split_count == 3


def test_absurd_dim_count_yields_no_tensors():
    kvs = [_kv_str("general.architecture", "llama"), _kv_u32("llama.block_count", 2)]
    tensors = [_tensor("blk.0.attn_q.weight", [1] * 9, 2)]
    m = parse_gguf_header(_blob(kvs, tensors))
    assert m.tensors == ()
    assert m.n_layers == 2


def test_implausible_tensor_count_yields_no_tensors():
    kvs = [_kv_str("general.architecture", "llama"), _kv_u32("llama.block_count", 2)]
    kvs_body = b"".join(kvs)
    blob = (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 2**64 - 1)
        + struct.pack("<Q", len(kvs))
        + kvs_body
    )
    m = parse_gguf_header(blob)
    assert m.tensors == ()
    assert m.n_layers == 2


def _hybrid_gguf():
    kvs = [
        _kv_str("general.architecture", "qwen3next"),
        _kv_u32("qwen3next.block_count", 8),
        _kv_u32("qwen3next.attention.head_count", 16),
        _kv_u32("qwen3next.attention.head_count_kv", 2),
        _kv_u32("qwen3next.embedding_length", 2048),
        _kv_u32("qwen3next.context_length", 262144),
        _kv_u32("qwen3next.full_attention_interval", 4),
        _kv_u32("qwen3next.attention.key_length", 256),
        _kv_u32("qwen3next.attention.value_length", 256),
        _kv_u32("qwen3next.ssm.conv_kernel", 4),
        _kv_u32("qwen3next.ssm.inner_size", 4096),
        _kv_u32("qwen3next.ssm.state_size", 128),
        _kv_u32("qwen3next.ssm.group_count", 16),
    ]
    body = b"".join(kvs)
    header = (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", len(kvs))
    )
    return header + body


def test_hybrid_and_state_keys():
    m = parse_gguf_header(_hybrid_gguf())
    assert m.full_attention_interval == 4
    assert (m.head_dim_k, m.head_dim_v) == (256, 256)
    assert (
        m.ssm_conv_kernel,
        m.ssm_inner_size,
        m.ssm_state_size,
        m.ssm_group_count,
    ) == (4, 4096, 128, 16)
    assert m.kv_layer_heads is None


def test_hybrid_keys_absent_are_none():
    m = parse_gguf_header(_synthetic_gguf())
    assert m.full_attention_interval is None and m.head_dim_k is None
    assert m.ssm_inner_size is None and m.kv_layer_heads is None


def test_per_layer_kv_heads_kept_raw():
    m = parse_gguf_header(_arr_head_gguf())
    assert m.kv_layer_heads == (8, 8, 4, 4)
    assert m.n_head_kv == 8


def test_per_layer_feed_forward_lengths_kept_beside_the_collapsed_one():
    """A header naming one feed-forward width per layer keeps the array as
    ff_layers, with n_ff still the collapsed maximum; a scalar width leaves
    ff_layers empty."""
    kvs = [
        _kv_str("general.architecture", "nemotron_h"),
        _kv_u32("nemotron_h.block_count", 4),
        _kv_u32("nemotron_h.attention.head_count", 32),
        _kv_arr_u32("nemotron_h.attention.head_count_kv", [0, 0, 8, 0]),
        _kv_arr_u32("nemotron_h.feed_forward_length", [0, 11008, 11008, 0]),
        _kv_u32("nemotron_h.embedding_length", 4096),
    ]
    header = (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", len(kvs))
    )
    m = parse_gguf_header(header + b"".join(kvs))
    assert m.ff_layers == (0, 11008, 11008, 0)
    assert m.n_ff == 11008
    assert parse_gguf_header(_arr_head_gguf()).ff_layers is None


def test_nested_array_head_count_kv_yields_none_without_raising():
    kvs = [
        _kv_str("general.architecture", "x"),
        _kv_u32("x.block_count", 3),
        _kv_u32("x.attention.head_count", 8),
        _kv_arr_arr_u32("x.attention.head_count_kv", [[2], [0], [2]]),
        _kv_u32("x.embedding_length", 64),
    ]
    header = (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", len(kvs))
    )
    m = parse_gguf_header(header + b"".join(kvs))
    assert m.kv_layer_heads is None
