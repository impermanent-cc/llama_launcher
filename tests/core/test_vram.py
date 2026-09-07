from types import SimpleNamespace

from llama_launcher.core.gguf import GgufMeta, TensorInfo
from llama_launcher.core.vram import (
    CARD_OVERHEAD_BYTES,
    COMPUTE_TERMS,
    ENGINE_COMPUTE_SCALE,
    CardEstimate,
    RamEstimate,
    bytes_per_elem,
    compute_bytes,
    effective_ctx_size,
    engine_scaled,
    estimate_memory,
    fits,
    host_compute_bytes,
    kv_cache_bytes,
    kv_layer_mask,
    logits_bytes,
    pooled_fit,
    recurrent_layer_mask,
    recurrent_state_bytes,
    slot_count,
    window_layer_mask,
    window_tokens,
)

MIB = 1024 * 1024


def _dense_meta(n_layers=4, tensors=True):
    ts = []
    if tensors:
        ts.append(TensorInfo("token_embd.weight", 1, 0, 100 * MIB))
        for i in range(n_layers):
            ts.append(TensorInfo(f"blk.{i}.attn_q.weight", 1, 0, 10 * MIB))
            ts.append(TensorInfo(f"blk.{i}.ffn_up.weight", 1, 0, 40 * MIB))
        ts.append(TensorInfo("output.weight", 1, 0, 50 * MIB))
    return GgufMeta(
        arch="llama",
        n_layers=n_layers,
        n_head=8,
        n_head_kv=8,
        n_embd=64,
        ctx_train=4096,
        n_ff=256,
        n_vocab=1000,
        tensors=tuple(ts),
    )


def _meta(**kw):
    base = dict(
        n_layers=2,
        n_head=8,
        n_head_kv=4,
        n_embd=64,
        ctx_train=4096,
        n_vocab=0,
        n_ff=0,
        n_ff_exp=None,
        n_expert_used=None,
        sliding_window=None,
        tensors=(),
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _hybrid_meta(**kw):
    base = dict(
        n_layers=40,
        n_head=16,
        n_head_kv=2,
        n_embd=2048,
        ctx_train=262144,
        n_vocab=248320,
        n_ff=0,
        n_ff_exp=1024,
        n_expert_used=8,
        expert_count=256,
        sliding_window=None,
        tensors=(),
        full_attention_interval=4,
        kv_layer_heads=None,
        head_dim_k=256,
        head_dim_v=256,
        ssm_conv_kernel=4,
        ssm_inner_size=4096,
        ssm_state_size=128,
        ssm_group_count=16,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_bytes_per_elem():
    assert bytes_per_elem("f16") == 2.0
    assert bytes_per_elem("q8_0") == 1.0625
    assert bytes_per_elem("unknown") == 2.0  # safe default


def test_kv_cache_bytes_f16():
    # 2 layers, 4 kv-heads, head_dim 8, ctx 100, f16 (2 bytes), K and V
    per = 2 * 100 * 4 * 8
    assert kv_cache_bytes(2, 4, 8, 100, "f16", "f16") == int(per * 2 + per * 2)


def test_fits():
    assert fits(100, 200) == (True, 100)
    ok, margin = fits(300, 200)
    assert ok is False and margin == -100


def test_fits_boundary_zero_margin():
    """When estimate equals free exactly, margin==0 and fits==True."""
    assert fits(200, 200) == (True, 0)


def test_available_free_bytes_sums_across_gpus_when_split():
    from llama_launcher.core.vram import available_free_bytes

    # A model split across GPUs (default split-mode) sees the COMBINED free
    # VRAM, not just the largest single card: a 16+12 GB rig with 14.7 + 7.3
    # GiB free offers ~22 GiB, so a 20.2 GiB model fits.
    gib = 1024**3
    free = [int(14.7 * gib), int(7.3 * gib)]
    assert available_free_bytes(free, "layer", 0) == sum(free)
    assert available_free_bytes(free, "row", 0) == sum(free)
    assert available_free_bytes(free, "tensor", 0) == sum(free)


def test_available_free_bytes_uses_main_gpu_when_split_none():
    from llama_launcher.core.vram import available_free_bytes

    gib = 1024**3
    free = [int(14.7 * gib), int(7.3 * gib)]
    # split-mode none puts the whole model on one GPU (main-gpu index).
    assert available_free_bytes(free, "none", 0) == free[0]
    assert available_free_bytes(free, "none", 1) == free[1]
    # out-of-range main-gpu falls back to the first card, not a crash.
    assert available_free_bytes(free, "none", 5) == free[0]


def test_available_free_bytes_empty():
    from llama_launcher.core.vram import available_free_bytes

    assert available_free_bytes([], "layer", 0) == 0


def test_pooled_fit_sums_vram_and_ram_and_reports_margin():
    gb = 1024**3
    r = pooled_fit(120 * gb, [("vram", 48 * gb), ("ram", 96 * gb)])
    assert r.fits is True
    assert r.vram_bytes == 48 * gb and r.ram_bytes == 96 * gb
    assert r.total_bytes == 144 * gb and r.margin == 24 * gb


def test_pooled_fit_does_not_fit_is_negative_margin():
    gb = 1024**3
    r = pooled_fit(200 * gb, [("vram", 48 * gb), ("ram", 96 * gb)])
    assert r.fits is False and r.margin == -56 * gb


def test_router_fit_sums_k_largest():
    from llama_launcher.core.vram import router_fit_summary

    s = router_fit_summary([10, 30, 20], models_max=2, free_bytes_per_gpu=[60])
    assert s.est_bytes == 50  # 30 + 20, the two largest
    assert s.models_counted == 2 and s.models_total == 3
    assert s.fits and s.margin == 10


def test_router_fit_models_max_zero_is_unlimited():
    from llama_launcher.core.vram import router_fit_summary

    s = router_fit_summary([10, 30, 20], models_max=0, free_bytes_per_gpu=[50])
    assert s.est_bytes == 60 and not s.fits and s.models_counted == 3


def test_router_fit_none_when_unknowable():
    from llama_launcher.core.vram import router_fit_summary

    assert router_fit_summary([], models_max=4, free_bytes_per_gpu=[10**9]) is None
    assert router_fit_summary([0, 0], models_max=4, free_bytes_per_gpu=[10**9]) is None
    assert router_fit_summary([100], models_max=4, free_bytes_per_gpu=[]) is None


# -- effective_ctx_size: the context the KV estimate should use -------------


def test_effective_ctx_size_prefers_an_explicit_ctx_size():
    s = {"ctx-size": 8192, "kv-unified-per-slot": 4096, "parallel": 4}
    assert effective_ctx_size(s, "llama.cpp") == 8192


def test_effective_ctx_size_multiplies_slots_by_the_per_slot_limit():
    s = {"kv-unified-per-slot": 4096, "parallel": 4}
    assert effective_ctx_size(s, "llama.cpp") == 16384


def test_effective_ctx_size_is_unknown_when_slots_are_auto():
    # --parallel -1 means the server picks the slot count at load time.
    s = {"kv-unified-per-slot": 4096, "parallel": -1}
    assert effective_ctx_size(s, "llama.cpp") is None
    assert effective_ctx_size({"kv-unified-per-slot": 4096}, "llama.cpp") is None


def test_effective_ctx_size_without_the_per_slot_flag_is_the_ctx_size():
    assert effective_ctx_size({"ctx-size": 2048}, "llama.cpp") == 2048
    assert effective_ctx_size({}, "llama.cpp") is None


def test_effective_ctx_size_ignores_the_flag_on_an_engine_that_lacks_it():
    s = {"kv-unified-per-slot": 4096, "parallel": 4}
    assert effective_ctx_size(s, "ik_llama.cpp") is None


def test_effective_ctx_size_rejects_values_that_are_not_slot_counts():
    neg = {"kv-unified-per-slot": -1, "parallel": 4}
    assert effective_ctx_size(neg, "llama.cpp") is None
    bool_parallel = {"kv-unified-per-slot": 4096, "parallel": True}
    assert effective_ctx_size(bool_parallel, "llama.cpp") is None
    zero_ctx = {"ctx-size": "0", "kv-unified-per-slot": 4096, "parallel": 4}
    assert effective_ctx_size(zero_ctx, "llama.cpp") == 16384


# compute_bytes and estimate_memory: placement-aware estimate


def test_compute_bytes_terms():
    m = _dense_meta()
    fa = compute_bytes(m, ubatch=512, ctx=4096, flash_attn=True)
    per_token = (
        COMPUTE_TERMS["logits"] * 1000
        + COMPUTE_TERMS["ffn"] * 256
        + COMPUTE_TERMS["residual"] * 64
        + COMPUTE_TERMS["vocab"] * 1000
    )
    assert fa == int(4 * 512 * per_token)
    no_fa = compute_bytes(m, ubatch=512, ctx=4096, flash_attn=False)
    assert no_fa == fa + int(4 * 512 * COMPUTE_TERMS["attn_scores"] * 8 * 4096)
    assert compute_bytes(m, ubatch=1024, ctx=4096, flash_attn=True) == 2 * fa


def test_compute_bytes_moe_uses_active_experts():
    m = GgufMeta(
        arch="moe",
        n_layers=1,
        n_head=1,
        n_head_kv=1,
        n_embd=8,
        n_ff=100,
        n_ff_exp=10,
        n_expert_used=4,
        n_vocab=0,
    )
    assert compute_bytes(m, ubatch=1, ctx=1, flash_attn=True) == int(
        4 * (COMPUTE_TERMS["ffn"] * 40 + COMPUTE_TERMS["residual"] * 8)
    )


def test_estimate_memory_all_on_one_card():
    m = _dense_meta()
    e = estimate_memory(
        m,
        0,
        settings={"ctx-size": 1024, "flash-attn": "on"},
        engine="llama.cpp",
        free_bytes_per_gpu=[16 * 1024 * MIB],
    )
    assert len(e.cards) == 1
    c = e.cards[0]
    assert c.weights == 4 * 50 * MIB + 50 * MIB
    assert c.kv == 4 * 1024 * 8 * 8 * 2 * 2
    assert c.overhead == CARD_OVERHEAD_BYTES
    assert c.compute == compute_bytes(m, ubatch=512, ctx=1024, flash_attn=True)
    assert e.ram.weights == 100 * MIB
    assert e.ram.kv == 0
    assert e.ram.buffers == 1000 * 4 * 4 + host_compute_bytes(
        m, ubatch=512, ctx=1024, flash_attn=True
    )
    assert e.ctx == 1024 and e.kv_upper_bound is False


def test_estimate_memory_n_cpu_ffn_moves_bytes_to_ram():
    m = _dense_meta()
    e = estimate_memory(
        m,
        0,
        settings={"ctx-size": 1024, "n-cpu-ffn": 2},
        engine="llama.cpp",
        free_bytes_per_gpu=[1],
    )
    assert e.cards[0].weights == 2 * 50 * MIB + 2 * 10 * MIB + 50 * MIB
    assert e.ram.weights == 100 * MIB + 2 * 40 * MIB


def test_estimate_memory_raw_args_override_settings():
    m = _dense_meta()
    a = estimate_memory(
        m,
        0,
        settings={"ctx-size": 1024, "n-cpu-ffn": 2},
        engine="llama.cpp",
        free_bytes_per_gpu=[1],
    )
    b = estimate_memory(
        m,
        0,
        settings={"ctx-size": 1024, "n-cpu-ffn": 2},
        engine="llama.cpp",
        free_bytes_per_gpu=[1],
        raw_args="--n-cpu-ffn 4",
    )
    assert b.ram.weights > a.ram.weights


def test_estimate_memory_two_cards_split_and_kv_follow_layers():
    m = _dense_meta()
    e = estimate_memory(
        m,
        0,
        settings={"ctx-size": 1024, "tensor-split": "50,50"},
        engine="llama.cpp",
        free_bytes_per_gpu=[1, 1],
    )
    assert len(e.cards) == 2
    assert e.cards[0].weights == 3 * 50 * MIB
    assert e.cards[1].weights == 50 * MIB + 50 * MIB
    per_layer_kv = 1024 * 8 * 8 * 2 * 2
    assert e.cards[0].kv == 3 * per_layer_kv and e.cards[1].kv == per_layer_kv
    assert e.cards[0].overhead == CARD_OVERHEAD_BYTES
    assert e.cards[1].overhead == CARD_OVERHEAD_BYTES


def test_estimate_memory_override_promoted_layer_weights_are_not_lost():
    """A layer --n-gpu-layers keeps in RAM but --override-tensor moves onto
    a card lands its bytes on a card, not nowhere: card weights plus RAM
    weights still sum to every tensor's bytes."""
    m = _dense_meta()
    e = estimate_memory(
        m,
        0,
        settings={
            "ctx-size": 1024,
            "n-gpu-layers": 3,
            "tensor-split": "50,50",
            "override-tensor": r"blk\.1\.=CUDA0",
        },
        engine="llama.cpp",
        free_bytes_per_gpu=[1, 1],
    )
    total_weights = 100 * MIB + 4 * (10 * MIB + 40 * MIB) + 50 * MIB
    assert sum(c.weights for c in e.cards) + e.ram.weights == total_weights


def test_estimate_memory_row_split_puts_compute_on_main_card_only():
    m = _dense_meta()
    e = estimate_memory(
        m,
        0,
        settings={
            "ctx-size": 1024,
            "tensor-split": "50,50",
            "split-mode": "row",
            "main-gpu": 1,
        },
        engine="llama.cpp",
        free_bytes_per_gpu=[1, 1],
    )
    assert e.cards[0].weights > 0 and e.cards[1].weights > 0
    assert e.cards[0].compute == 0
    assert e.cards[1].compute == compute_bytes(m, ubatch=512, ctx=1024, flash_attn=True)


def test_estimate_memory_no_kv_offload_and_partial_ngl_put_kv_in_ram():
    m = _dense_meta()
    e = estimate_memory(
        m,
        0,
        settings={"ctx-size": 1024, "no-kv-offload": True},
        engine="llama.cpp",
        free_bytes_per_gpu=[1],
    )
    assert e.cards[0].kv == 0 and e.ram.kv == 4 * 1024 * 8 * 8 * 2 * 2
    e = estimate_memory(
        m,
        0,
        settings={"ctx-size": 1024, "n-gpu-layers": 3},
        engine="llama.cpp",
        free_bytes_per_gpu=[1],
    )
    assert e.ram.kv == 2 * 1024 * 8 * 8 * 2 * 2


def test_estimate_memory_ik_auto_offloads_nothing():
    m = _dense_meta()
    e = estimate_memory(
        m,
        0,
        settings={"ctx-size": 1024},
        engine="ik_llama.cpp",
        free_bytes_per_gpu=[1],
    )
    assert e.cards[0].weights == 0 and e.cards[0].kv == 0
    assert e.cards[0].compute == 0
    assert e.cards[0].overhead == CARD_OVERHEAD_BYTES
    assert e.ram.weights == 100 * MIB + 4 * 50 * MIB + 50 * MIB


def test_estimate_memory_fallback_without_tensor_table_uses_file_size():
    m = _dense_meta(tensors=False)
    e = estimate_memory(
        m,
        7 * MIB,
        settings={"ctx-size": 1024},
        engine="llama.cpp",
        free_bytes_per_gpu=[1],
    )
    assert e.cards[0].weights == 7 * MIB and e.ram.weights == 0


def test_estimate_memory_draft_and_mmproj():
    m = _dense_meta()
    d = _dense_meta(n_layers=2)
    e = estimate_memory(
        m,
        0,
        settings={"ctx-size": 1024},
        engine="llama.cpp",
        free_bytes_per_gpu=[1],
        draft_meta=d,
        mmproj_bytes=9 * MIB,
    )
    base = estimate_memory(
        m,
        0,
        settings={"ctx-size": 1024},
        engine="llama.cpp",
        free_bytes_per_gpu=[1],
    )
    assert (
        e.cards[0].weights == base.cards[0].weights + 2 * 50 * MIB + 50 * MIB + 9 * MIB
    )
    assert e.cards[0].kv == base.cards[0].kv + 2 * 1024 * 8 * 8 * 2 * 2
    off = estimate_memory(
        m,
        0,
        settings={"ctx-size": 1024, "no-mmproj-offload": True},
        engine="llama.cpp",
        free_bytes_per_gpu=[1],
        mmproj_bytes=9 * MIB,
    )
    assert off.cards[0].weights == base.cards[0].weights
    assert off.ram.weights == base.ram.weights + 9 * MIB


def test_estimate_memory_draft_cpu_moe_twin_applies_to_draft_only():
    m = _dense_meta()
    d = _dense_meta(n_layers=2)
    d = GgufMeta(
        **{
            **d.__dict__,
            "tensors": tuple(
                TensorInfo(t.name.replace("ffn_up", "ffn_up_exps"), 1, 0, t.nbytes)
                for t in d.tensors
            ),
        }
    )
    e = estimate_memory(
        m,
        0,
        settings={"ctx-size": 1024, "spec-draft-cpu-moe": True},
        engine="llama.cpp",
        free_bytes_per_gpu=[1],
        draft_meta=d,
    )
    plain = estimate_memory(
        m,
        0,
        settings={"ctx-size": 1024},
        engine="llama.cpp",
        free_bytes_per_gpu=[1],
        draft_meta=d,
    )
    assert plain.cards[0].weights - e.cards[0].weights == 2 * 40 * MIB


def test_estimate_memory_swa_flags_upper_bound():
    m = GgufMeta(**{**_dense_meta().__dict__, "sliding_window": 512})
    assert (
        estimate_memory(
            m,
            0,
            settings={"ctx-size": 1024},
            engine="llama.cpp",
            free_bytes_per_gpu=[1],
        ).kv_upper_bound
        is True
    )
    assert (
        estimate_memory(
            m,
            0,
            settings={"ctx-size": 1024, "swa-full": True},
            engine="llama.cpp",
            free_bytes_per_gpu=[1],
        ).kv_upper_bound
        is False
    )


def test_estimate_memory_unknowable_returns_none():
    assert (
        estimate_memory(
            None, 5, settings={}, engine="llama.cpp", free_bytes_per_gpu=[1]
        )
        is None
    )
    assert (
        estimate_memory(
            GgufMeta(arch="x"),
            5,
            settings={},
            engine="llama.cpp",
            free_bytes_per_gpu=[1],
        )
        is None
    )


def test_estimate_memory_non_numeric_main_gpu_puts_mmproj_on_card_0():
    """A hand-edited or raw-arg main-gpu value that is not a number falls
    back to card 0 instead of raising."""
    m = _dense_meta()
    base = estimate_memory(
        m,
        0,
        settings={"ctx-size": 1024},
        engine="llama.cpp",
        free_bytes_per_gpu=[1, 1],
    )
    e = estimate_memory(
        m,
        0,
        settings={"ctx-size": 1024, "main-gpu": "CUDA1"},
        engine="llama.cpp",
        free_bytes_per_gpu=[1, 1],
        mmproj_bytes=9 * MIB,
    )
    assert e.cards[0].weights == base.cards[0].weights + 9 * MIB
    assert e.cards[1].weights == base.cards[1].weights


def test_estimate_memory_fallback_without_tensor_table_splits_across_cards():
    """With no tensor table the fallback weight size is one blob that
    spreads across cards by --tensor-split, not by a single discrete
    layer position."""
    m = _dense_meta(tensors=False)
    e = estimate_memory(
        m,
        8 * MIB,
        settings={"ctx-size": 1024, "tensor-split": "50,50"},
        engine="llama.cpp",
        free_bytes_per_gpu=[1, 1],
    )
    assert e.cards[0].weights == 4 * MIB and e.cards[1].weights == 4 * MIB


def test_estimate_memory_gqa_head_dim_uses_query_heads():
    """KV-cache head_dim is n_embd // n_head (the query head count), not
    n_head_kv, on a grouped-query-attention model."""
    m = GgufMeta(
        arch="llama",
        n_layers=1,
        n_head=8,
        n_head_kv=4,
        n_embd=64,
        ctx_train=4096,
        n_ff=64,
        n_vocab=10,
        tensors=(TensorInfo("output.weight", 1, 0, 10 * MIB),),
    )
    e = estimate_memory(
        m,
        0,
        settings={"ctx-size": 100},
        engine="llama.cpp",
        free_bytes_per_gpu=[10**9],
    )
    head_dim = 64 // 8
    expected_kv = kv_cache_bytes(1, 4, head_dim, 100, "f16", "f16")
    assert e.cards[0].kv == expected_kv


def test_estimate_memory_clamps_ubatch_to_batch():
    """A configured ubatch above the batch size is clamped to the batch
    size, the cap llama.cpp itself applies to the micro-batch."""
    m = _dense_meta()
    a = estimate_memory(
        m,
        0,
        settings={"ctx-size": 4096, "batch-size": 1024, "ubatch-size": 4096},
        engine="llama.cpp",
        free_bytes_per_gpu=[10**9],
    )
    b = estimate_memory(
        m,
        0,
        settings={"ctx-size": 4096, "batch-size": 1024, "ubatch-size": 1024},
        engine="llama.cpp",
        free_bytes_per_gpu=[10**9],
    )
    assert a.cards[0].compute == b.cards[0].compute


def test_estimate_memory_draft_no_bytes_on_card_adds_no_compute():
    """A card that holds none of the draft model's weights or KV gets no
    draft compute-buffer charge."""
    m = _dense_meta()
    d = _dense_meta(n_layers=2)
    e = estimate_memory(
        m,
        0,
        settings={"ctx-size": 1024, "tensor-split": "100,0"},
        engine="llama.cpp",
        free_bytes_per_gpu=[1, 1],
        draft_meta=d,
    )
    assert e.cards[1].weights == 0 and e.cards[1].kv == 0 and e.cards[1].compute == 0
    expected = compute_bytes(m, ubatch=512, ctx=1024, flash_attn=True) + compute_bytes(
        d, ubatch=512, ctx=1024, flash_attn=True
    )
    assert e.cards[0].compute == expected


def test_estimate_memory_draft_cache_type_draft_shrinks_draft_kv_only():
    """cache-type-k-draft/v-draft quantize only the draft model's KV cache;
    the main model's cache-type-k/v and resulting KV size are unchanged."""
    m = _dense_meta()
    d = _dense_meta(n_layers=2)
    f16 = estimate_memory(
        m,
        0,
        settings={"ctx-size": 1024},
        engine="llama.cpp",
        free_bytes_per_gpu=[1],
        draft_meta=d,
    )
    q8 = estimate_memory(
        m,
        0,
        settings={
            "ctx-size": 1024,
            "cache-type-k-draft": "q8_0",
            "cache-type-v-draft": "q8_0",
        },
        engine="llama.cpp",
        free_bytes_per_gpu=[1],
        draft_meta=d,
    )
    assert q8.cards[0].kv < f16.cards[0].kv
    main_only_f16 = estimate_memory(
        m,
        0,
        settings={"ctx-size": 1024},
        engine="llama.cpp",
        free_bytes_per_gpu=[1],
    )
    main_only_q8 = estimate_memory(
        m,
        0,
        settings={
            "ctx-size": 1024,
            "cache-type-k-draft": "q8_0",
            "cache-type-v-draft": "q8_0",
        },
        engine="llama.cpp",
        free_bytes_per_gpu=[1],
    )
    assert main_only_f16.cards[0].kv == main_only_q8.cards[0].kv


def test_kv_layer_mask_interval_array_and_default():
    """A full-attention interval marks every interval-th layer; a per-layer
    KV head array marks the layers with a non-zero count; a header with
    neither marks every layer."""
    assert kv_layer_mask(_hybrid_meta(), 8) == (False, False, False, True) * 2
    m = _meta(kv_layer_heads=(2, 0, 2))
    assert kv_layer_mask(m, 3) == (True, False, True)
    assert kv_layer_mask(_meta(), 2) == (True, True)


def test_recurrent_state_bytes_matches_the_server_checkpoint():
    """One recurrent layer's f32 state for one request slot is the
    convolution state plus the state matrix; a header with no ssm sizes
    carries none."""
    per_layer = recurrent_state_bytes(_hybrid_meta())
    assert per_layer == 4 * (3 * (4096 + 2 * 16 * 128) + 128 * 4096)
    assert abs(30 * per_layer / MIB - 62.8) < 0.2
    assert recurrent_state_bytes(_meta()) == 0


def test_hybrid_kv_charged_to_kv_layers_only_with_header_head_size():
    """KV is charged to the 10 full-attention layers of a 40-layer hybrid at
    the header's key and value lengths; the other 30 carry recurrent state,
    one copy per request slot, on the cards that hold them, the default 32
    checkpoints of that state land in RAM, and there is no upper-bound
    label."""
    est = estimate_memory(
        _hybrid_meta(),
        20 * 1024**3,
        settings={"ctx-size": 131072, "n-gpu-layers": "all", "flash-attn": "on"},
        engine="llama.cpp",
        free_bytes_per_gpu=[15 * 1024**3, 12 * 1024**3],
    )
    kv_total = sum(c.kv for c in est.cards) + est.ram.kv
    assert kv_total == 10 * 131072 * 2 * 2 * 256 * 2
    assert kv_total == 2560 * MIB
    state_total = sum(c.state for c in est.cards) + est.ram.state
    per_slot = 30 * recurrent_state_bytes(_hybrid_meta())
    assert state_total == 4 * per_slot
    assert est.ram.state == 0
    assert est.ram.checkpoints == 32 * state_total
    assert est.kv_upper_bound is False


def test_recurrent_state_scales_with_the_slot_count():
    """The recurrent state is sized once per request slot: --parallel 1 and
    the ik default of one slot carry a quarter of what llama.cpp's default
    of four carries, and the checkpoints follow the same state."""
    common = dict(
        settings={"ctx-size": 4096, "n-gpu-layers": "all"},
        free_bytes_per_gpu=[15 * 1024**3],
    )
    four = estimate_memory(_hybrid_meta(), 20 * 1024**3, engine="llama.cpp", **common)
    one = estimate_memory(
        _hybrid_meta(),
        20 * 1024**3,
        settings={"ctx-size": 4096, "n-gpu-layers": "all", "parallel": 1},
        engine="llama.cpp",
        free_bytes_per_gpu=[15 * 1024**3],
    )
    ik = estimate_memory(_hybrid_meta(), 20 * 1024**3, engine="ik_llama.cpp", **common)
    per_slot = 30 * recurrent_state_bytes(_hybrid_meta())
    assert four.cards[0].state == 4 * per_slot
    assert one.cards[0].state == per_slot
    assert ik.cards[0].state == per_slot
    assert four.ram.checkpoints == 4 * one.ram.checkpoints
    assert one.ram.checkpoints == 32 * per_slot


def test_checkpoints_follow_the_setting_and_state_moves_to_ram_with_kv():
    """--ctx-checkpoints sets the checkpoint count, and --no-kv-offload moves
    the recurrent state to RAM with the KV cache; the checkpoints are in RAM
    either way."""
    est = estimate_memory(
        _hybrid_meta(),
        20 * 1024**3,
        settings={"ctx-size": 4096, "ctx-checkpoints": 2, "no-kv-offload": True},
        engine="llama.cpp",
        free_bytes_per_gpu=[15 * 1024**3],
    )
    assert est.cards[0].kv == 0
    assert est.cards[0].state == 0
    assert est.ram.state == 4 * 30 * recurrent_state_bytes(_hybrid_meta())
    assert est.ram.checkpoints == 2 * est.ram.state


def test_checkpoints_stay_in_ram_when_the_state_is_on_a_card():
    """The server keeps its context checkpoints as host vectors: no card
    carries a checkpoint byte, and RAM carries the count times the state of
    every recurrent layer even when every layer sits on a card."""
    est = estimate_memory(
        _hybrid_meta(),
        20 * 1024**3,
        settings={"ctx-size": 4096, "n-gpu-layers": "all"},
        engine="llama.cpp",
        free_bytes_per_gpu=[15 * 1024**3, 12 * 1024**3],
    )
    state_total = sum(c.state for c in est.cards)
    assert state_total > 0 and est.ram.state == 0
    assert est.ram.checkpoints == 32 * state_total
    assert not hasattr(est.cards[0], "checkpoints")


def test_card_and_ram_totals_include_state_and_checkpoints():
    """Recurrent state counts into the per-card total; state and checkpoints
    count into the RAM total."""
    c = CardEstimate(1, 2, 3, 4, state=5)
    assert c.total == 15
    r = RamEstimate(1, 2, 3, state=4, checkpoints=5)
    assert r.total == 15


def test_ctx_checkpoints_zero_charges_no_checkpoints():
    """--ctx-checkpoints 0, the catalog minimum, keeps no checkpoints: the
    recurrent state still counts, the checkpoints term is zero."""
    est = estimate_memory(
        _hybrid_meta(),
        20 * 1024**3,
        settings={"ctx-size": 4096, "ctx-checkpoints": 0},
        engine="llama.cpp",
        free_bytes_per_gpu=[15 * 1024**3, 12 * 1024**3],
    )
    assert est.ram.checkpoints == 0
    state_total = sum(c.state for c in est.cards) + est.ram.state
    assert state_total == 4 * 30 * recurrent_state_bytes(_hybrid_meta())


def test_full_attention_interval_without_state_sizes_charges_every_layer():
    """An interval the header names without recurrent-state sizes is not
    trusted: every layer is charged KV, and no state or checkpoints."""
    meta = _hybrid_meta(ssm_conv_kernel=None, ssm_inner_size=None, ssm_state_size=None)
    assert kv_layer_mask(meta, 8) == (True,) * 8
    est = estimate_memory(
        meta,
        20 * 1024**3,
        settings={"ctx-size": 131072, "n-gpu-layers": "all", "flash-attn": "on"},
        engine="llama.cpp",
        free_bytes_per_gpu=[15 * 1024**3, 12 * 1024**3],
    )
    kv_total = sum(c.kv for c in est.cards) + est.ram.kv
    assert kv_total == 40 * 131072 * 2 * 2 * 256 * 2
    assert sum(c.state for c in est.cards) + est.ram.state == 0
    assert est.ram.checkpoints == 0


def test_model_total_includes_state_and_checkpoints():
    """The figure a pool or router readout spreads across its budget counts
    the recurrent state and checkpoints on every device."""
    est = estimate_memory(
        _hybrid_meta(),
        20 * 1024**3,
        settings={"ctx-size": 4096, "n-gpu-layers": "all"},
        engine="llama.cpp",
        free_bytes_per_gpu=[15 * 1024**3, 12 * 1024**3],
    )
    expected = sum(c.weights + c.kv + c.state for c in est.cards) + (
        est.ram.weights + est.ram.kv + est.ram.state + est.ram.checkpoints
    )
    assert est.model_total == expected
    assert sum(c.state for c in est.cards) > 0


def test_slot_count_setting_and_engine_defaults():
    """Request slots come from --parallel when it is a positive number, and
    otherwise from the engine default: four on llama.cpp, one on
    ik_llama.cpp."""
    assert slot_count({"parallel": 2}, "llama.cpp") == 2
    assert slot_count({}, "llama.cpp") == 4
    assert slot_count({"parallel": -1}, "llama.cpp") == 4
    assert slot_count({}, "ik_llama.cpp") == 1
    assert slot_count({"parallel": True}, "ik_llama.cpp") == 1


def test_host_compute_bytes_is_the_card_formula_at_the_host_term():
    """The host compute buffer is the host term of the table times the
    per-card formula without its logits or vocabulary term, and nothing at
    all when that formula is empty."""
    m = _meta(n_embd=64, n_vocab=1000)
    base = compute_bytes(
        m, ubatch=512, ctx=4096, flash_attn=True, logits=False, vocab=False
    )
    assert host_compute_bytes(m, ubatch=512, ctx=4096, flash_attn=True) == int(
        COMPUTE_TERMS["host"] * base
    )
    assert (
        host_compute_bytes(_meta(n_embd=0), ubatch=512, ctx=4096, flash_attn=True) == 0
    )


def test_recurrent_term_is_zero_without_an_inner_size_and_scales_with_it():
    """The recurrent term charges the header's ssm inner size per micro-batch
    token in both the card and the host compute buffer, and charges nothing
    on a header that carries no inner size."""
    plain = _meta(n_embd=64)
    small = _meta(n_embd=64, ssm_inner_size=128)
    large = _meta(n_embd=64, ssm_inner_size=256)
    term = int(4 * 512 * COMPUTE_TERMS["ssm"] * 128)
    card = compute_bytes(plain, ubatch=512, ctx=4096, flash_attn=True)
    assert compute_bytes(small, ubatch=512, ctx=4096, flash_attn=True) - card == term
    assert (
        compute_bytes(large, ubatch=512, ctx=4096, flash_attn=True) - card == 2 * term
    )
    host = host_compute_bytes(plain, ubatch=512, ctx=4096, flash_attn=True)
    host_term = int(COMPUTE_TERMS["host"] * term)
    assert (
        host_compute_bytes(small, ubatch=512, ctx=4096, flash_attn=True) - host
        == host_term
    )
    assert (
        host_compute_bytes(large, ubatch=512, ctx=4096, flash_attn=True) - host
        == 2 * host_term
    )


def test_ik_engine_takes_its_share_of_the_compute_and_host_figures():
    """A profile on ik_llama.cpp carries the engine table's fraction of the
    card compute buffer and the host buffer that mainline llama.cpp carries
    for the same model and settings."""
    m = _meta(n_embd=64, n_vocab=1000, ssm_inner_size=128)
    common = dict(
        settings={
            "ctx-size": 1024,
            "n-gpu-layers": "all",
            "flash-attn": "on",
            "parallel": 4,
        },
        free_bytes_per_gpu=[8 * 1024**3],
    )
    main = estimate_memory(m, 10**9, engine="llama.cpp", **common)
    ik = estimate_memory(m, 10**9, engine="ik_llama.cpp", **common)
    scale = ENGINE_COMPUTE_SCALE["ik_llama.cpp"]
    assert main.cards[0].compute > 0 and main.ram.host > 0
    assert ik.cards[0].compute == int(scale * main.cards[0].compute)
    # The output buffer is the same on both engines at the same slot count;
    # only the host compute part takes the scale.
    assert ik.ram.output == 1000 * 4 * 4 == main.ram.output
    assert ik.ram.host == int(scale * main.ram.host)
    assert ik.ram.buffers == 1000 * 4 * 4 + int(scale * main.ram.host)


def test_compute_bytes_logits_term_is_optional():
    """The logits term is charged only when asked for; every other term of
    the compute buffer is unchanged."""
    m = _dense_meta()
    with_logits = compute_bytes(m, ubatch=512, ctx=4096, flash_attn=True)
    without = compute_bytes(m, ubatch=512, ctx=4096, flash_attn=True, logits=False)
    assert with_logits - without == int(4 * 512 * COMPUTE_TERMS["logits"] * 1000)


def test_output_buffer_is_vocab_by_slots():
    """RAM buffers hold f32 logits for every slot plus the host compute
    buffer."""
    m = _meta(n_vocab=262144)
    est = estimate_memory(
        m,
        10**9,
        settings={"ctx-size": 2048},
        engine="llama.cpp",
        free_bytes_per_gpu=[8 * 1024**3],
    )
    assert est.ram.buffers == 262144 * 4 * 4 + host_compute_bytes(
        m, ubatch=512, ctx=2048, flash_attn=True
    )


def test_output_buffer_follows_the_parallel_setting():
    """The output buffer scales with the slot count --parallel asks for."""
    m = _meta(n_vocab=262144)
    est = estimate_memory(
        m,
        10**9,
        settings={"ctx-size": 2048, "parallel": 8},
        engine="llama.cpp",
        free_bytes_per_gpu=[8 * 1024**3],
    )
    assert est.ram.buffers == 262144 * 4 * 8 + host_compute_bytes(
        m, ubatch=512, ctx=2048, flash_attn=True
    )


def test_logits_charged_to_the_output_card_only():
    """Under a two-card split every used card carries the compute buffer
    without its logits term, and only the card holding the output layer
    carries the logits on top."""
    m = _meta(n_vocab=100000, n_layers=4, n_ff=256)
    est = estimate_memory(
        m,
        10**9,
        settings={"ctx-size": 2048, "tensor-split": "1,1"},
        engine="llama.cpp",
        free_bytes_per_gpu=[8 * 1024**3, 8 * 1024**3],
    )
    base = compute_bytes(m, ubatch=512, ctx=2048, flash_attn=True, logits=False)
    with_logits = compute_bytes(m, ubatch=512, ctx=2048, flash_attn=True, logits=True)
    assert sorted(c.compute for c in est.cards) == [base, with_logits]
    assert est.cards[1].compute == with_logits


def test_logits_stay_in_ram_when_the_output_layer_does():
    """An override that keeps the output tensor in host RAM keeps the
    logits term there too: no card carries it, it joins the host compute
    part where the host computes it, the output part stays the buffer the
    server hands back, and the RAM buffers grow by exactly that term."""
    m = _dense_meta()
    est = estimate_memory(
        m,
        0,
        settings={"ctx-size": 2048, "override-tensor": "output=CPU"},
        engine="llama.cpp",
        free_bytes_per_gpu=[8 * 1024**3],
    )
    base = compute_bytes(m, ubatch=512, ctx=2048, flash_attn=True, logits=False)
    with_logits = compute_bytes(m, ubatch=512, ctx=2048, flash_attn=True, logits=True)
    assert est.cards[0].compute == base
    assert est.ram.output == 1000 * 4 * 4
    assert est.ram.host == (
        host_compute_bytes(m, ubatch=512, ctx=2048, flash_attn=True)
        + (with_logits - base)
    )
    assert est.ram.buffers == est.ram.host + est.ram.output


def test_row_split_charges_the_logits_on_the_main_card():
    """Row mode counts the compute buffer once, on --main-gpu, and the
    logits term goes to that same card."""
    m = _dense_meta()
    est = estimate_memory(
        m,
        0,
        settings={"ctx-size": 2048, "split-mode": "row", "main-gpu": 1},
        engine="llama.cpp",
        free_bytes_per_gpu=[8 * 1024**3, 8 * 1024**3],
    )
    assert est.cards[0].compute == 0
    assert est.cards[1].compute == compute_bytes(
        m, ubatch=512, ctx=2048, flash_attn=True
    )


def test_draft_logits_follow_the_draft_output_layer():
    """The draft model's logits term follows its own placement: on the
    card holding its output layer, in RAM when that layer stays there."""
    main = _dense_meta()
    draft = _dense_meta(n_layers=2)
    on_card = estimate_memory(
        main,
        0,
        settings={"ctx-size": 1024},
        engine="llama.cpp",
        free_bytes_per_gpu=[8 * 1024**3],
        draft_meta=draft,
        draft_weights=0,
    )
    in_ram = estimate_memory(
        main,
        0,
        settings={"ctx-size": 1024, "spec-draft-override-tensor": "output=CPU"},
        engine="llama.cpp",
        free_bytes_per_gpu=[8 * 1024**3],
        draft_meta=draft,
        draft_weights=0,
    )
    delta = compute_bytes(draft, ubatch=512, ctx=1024, flash_attn=True) - compute_bytes(
        draft, ubatch=512, ctx=1024, flash_attn=True, logits=False
    )
    assert delta > 0
    assert on_card.cards[0].compute - in_ram.cards[0].compute == delta
    assert in_ram.ram.buffers - on_card.ram.buffers == delta


def test_ffn_term_is_at_least_one_f32_copy():
    """The ffn term never estimates less than the physical floor: one f32
    copy of the FFN intermediate per micro-batch token."""
    assert COMPUTE_TERMS["ffn"] >= 1.0


def _mamba_meta(**kw):
    """A purely recurrent header: state sizes, no attention heads, no
    full-attention interval and no per-layer array."""
    base = dict(
        n_layers=64,
        n_head=0,
        n_head_kv=0,
        n_embd=4096,
        ctx_train=131072,
        n_vocab=50280,
        n_ff=14336,
        n_ff_exp=None,
        n_expert_used=None,
        sliding_window=None,
        tensors=(),
        full_attention_interval=None,
        kv_layer_heads=None,
        ssm_conv_kernel=4,
        ssm_inner_size=8192,
        ssm_state_size=128,
        ssm_group_count=8,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_pure_recurrent_header_charges_no_kv_and_every_layer_state():
    """A header with recurrent-state sizes and no attention heads is purely
    recurrent: no layer holds a KV cache, every layer holds state on the
    card that holds it, and the checkpoints of all of it land in RAM."""
    meta = _mamba_meta()
    assert kv_layer_mask(meta, 64) == (False,) * 64
    est = estimate_memory(
        meta,
        8 * 1024**3,
        settings={"ctx-size": 8192, "n-gpu-layers": "all", "parallel": 1},
        engine="llama.cpp",
        free_bytes_per_gpu=[24 * 1024**3],
    )
    assert sum(c.kv for c in est.cards) + est.ram.kv == 0
    per_layer = recurrent_state_bytes(meta)
    assert per_layer == 4 * (3 * (8192 + 2 * 8 * 128) + 128 * 8192)
    assert est.cards[0].state == 64 * per_layer
    assert est.ram.state == 0
    assert est.ram.checkpoints == 32 * 64 * per_layer


def test_mlp_only_layers_of_a_three_type_hybrid_carry_neither_kv_nor_state():
    """On a header carrying per-layer KV head counts and per-layer
    feed-forward widths, a layer with no KV heads and a non-zero width is
    MLP-only: it holds no cache and no recurrent state, while a zero-width
    layer without KV heads holds state and an attention layer holds its
    cache."""
    meta = _meta(
        n_layers=4,
        n_head=32,
        n_head_kv=4,
        n_embd=4096,
        kv_layer_heads=(0, 0, 4, 0),
        ff_layers=(0, 11008, 11008, 0),
        ssm_conv_kernel=4,
        ssm_inner_size=4096,
        ssm_state_size=128,
        ssm_group_count=8,
    )
    mask = kv_layer_mask(meta, 4)
    assert mask == (False, False, True, False)
    assert recurrent_layer_mask(meta, mask) == (True, False, False, True)
    est = estimate_memory(
        meta,
        10**9,
        settings={"ctx-size": 1024, "n-gpu-layers": "all", "parallel": 1},
        engine="llama.cpp",
        free_bytes_per_gpu=[24 * 1024**3],
    )
    head_dim = 4096 // 32
    assert est.cards[0].kv == 1024 * 4 * head_dim * 2 * 2
    assert est.cards[0].state == 2 * recurrent_state_bytes(meta)
    assert est.ram.checkpoints == 32 * est.cards[0].state


def test_each_layer_is_priced_from_its_own_kv_head_count():
    """A per-layer KV head-count array sizes each layer's cache by its own
    count, not by the largest count in the array."""
    meta = _meta(
        n_layers=4,
        n_head=32,
        n_head_kv=8,
        n_embd=4096,
        kv_layer_heads=(8, 8, 4, 4),
    )
    est = estimate_memory(
        meta,
        10**9,
        settings={"ctx-size": 1024, "n-gpu-layers": "all"},
        engine="llama.cpp",
        free_bytes_per_gpu=[24 * 1024**3],
    )
    head_dim = 4096 // 32
    per_head_layer = 1024 * head_dim * 2 * 2
    assert est.cards[0].kv == (8 + 8 + 4 + 4) * per_head_layer
    assert est.cards[0].kv == 12582912


def test_logits_stay_in_ram_when_a_tableless_model_keeps_the_output_layer():
    """With no tensor table the whole model is one fallback blob placed at
    the output position, but the logits term follows the output layer's own
    device: ik_llama.cpp with --n-gpu-layers below the layer count keeps
    that layer in RAM, so the card carries the compute buffer without the
    logits and the host buffer carries them."""
    m = _dense_meta(tensors=False)
    est = estimate_memory(
        m,
        10**9,
        settings={"ctx-size": 2048, "n-gpu-layers": 3},
        engine="ik_llama.cpp",
        free_bytes_per_gpu=[8 * 1024**3],
    )
    base = compute_bytes(m, ubatch=512, ctx=2048, flash_attn=True, logits=False)
    logits = logits_bytes(m, 512)
    assert logits > 0
    assert est.cards[0].weights > 0
    assert est.cards[0].compute == engine_scaled(base, "ik_llama.cpp")
    assert est.ram.host == engine_scaled(
        host_compute_bytes(m, ubatch=512, ctx=2048, flash_attn=True), "ik_llama.cpp"
    ) + engine_scaled(logits, "ik_llama.cpp")


def test_logits_bytes_is_the_delta_the_compute_formula_carries():
    """The logits companion is exactly what the compute formula adds when it
    is asked for the logits term."""
    m = _dense_meta()
    with_logits = compute_bytes(m, ubatch=512, ctx=2048, flash_attn=True)
    without = compute_bytes(m, ubatch=512, ctx=2048, flash_attn=True, logits=False)
    assert with_logits - without == logits_bytes(m, 512)
    assert logits_bytes(m, 512) == int(4 * 512 * COMPUTE_TERMS["logits"] * 1000)


def _swa_meta(**kw):
    """Six layers, five windowed then one full, Gemma 4 style head sizes."""
    base = dict(
        n_layers=6,
        n_head=16,
        n_head_kv=8,
        kv_layer_heads=(8, 8, 8, 8, 8, 2),
        head_dim_k=512,
        head_dim_v=512,
        head_dim_k_swa=256,
        head_dim_v_swa=256,
        sliding_window=1024,
        sliding_window_pattern=(True, True, True, True, True, False),
        shared_kv_layers=0,
    )
    base.update(kw)
    return _meta(**base)


def _kv_of(meta, settings, engine="llama.cpp"):
    est = estimate_memory(
        meta,
        0,
        settings={"n-gpu-layers": "all", **settings},
        engine=engine,
        free_bytes_per_gpu=[1],
    )
    return est.cards[0].kv, est.kv_upper_bound


# One window layer at full context: 8 heads x 256 x 2 bytes x K and V
WINDOW_FULL = 8 * 256 * 2 * 2 * 32768
# The single full-attention layer: 2 heads x 512 x 2 bytes x K and V
FULL_LAYER = 2 * 512 * 2 * 2 * 32768


def test_window_layers_use_the_swa_head_size_under_swa_full():
    kv, label = _kv_of(_swa_meta(), {"ctx-size": 32768, "swa-full": True})
    assert kv == 5 * WINDOW_FULL + FULL_LAYER
    assert label is False


def test_window_layers_hold_window_plus_ubatch_padded_to_256():
    """Without --swa-full a window layer holds min(ctx per slot, window +
    ubatch) rounded up to 256 tokens per slot: 1024 + 512 = 1536 here."""
    kv, label = _kv_of(_swa_meta(), {"ctx-size": 32768, "parallel": 1})
    per_token = 8 * 256 * 2 * 2
    assert kv == 5 * per_token * 1536 + FULL_LAYER
    assert label is False


def test_window_tokens_rule():
    assert window_tokens(32768, 1024, 512, 1, False) == 1536
    assert window_tokens(32768, 1024, 512, 2, False) == 2 * 1536
    assert window_tokens(32768, 1024, 512, 2, True) == 2560
    assert window_tokens(2048, 512, 512, 4, False) == 4 * 512
    assert window_tokens(1000, 1024, 512, 1, False) == 1024
    assert window_tokens(300, 100, 100, 1, False) == 256
    # The unified branch caps the total at ctx: window * slots + ubatch
    # (4608) exceeds a 2048 context, so the pad settles at ctx itself.
    assert window_tokens(2048, 1024, 512, 4, True) == 2048
    # A context smaller than the slot count floors ctx // slots at 1 token,
    # which still pads up to a full 256-token slot.
    assert window_tokens(3, 100, 10, 8, False) == 8 * 256


def test_window_layer_mask_pads_a_short_pattern_with_false():
    """A pattern array shorter than the layer count leaves every layer past
    its end False rather than raising or repeating the pattern."""
    m = _swa_meta(sliding_window_pattern=(True, True))
    assert window_layer_mask(m, 6) == (True, True, False, False, False, False)


def test_unified_cache_sizes_the_window_once_across_slots():
    kv, _ = _kv_of(_swa_meta(), {"ctx-size": 32768, "parallel": 2, "kv-unified": True})
    per_token = 8 * 256 * 2 * 2
    assert kv == 5 * per_token * 2560 + FULL_LAYER


def test_pattern_less_sliding_window_stays_the_labelled_upper_bound():
    m = _swa_meta(sliding_window_pattern=None)
    kv, label = _kv_of(m, {"ctx-size": 32768})
    # Every layer at full context and at the full-attention head size.
    assert kv == 5 * (8 * 512 * 2 * 2 * 32768) + FULL_LAYER
    assert label is True
    _, label_full = _kv_of(m, {"ctx-size": 32768, "swa-full": True})
    assert label_full is False


def test_ik_prices_window_layers_at_full_context_and_keeps_the_label():
    kv, label = _kv_of(_swa_meta(), {"ctx-size": 32768}, engine="ik_llama.cpp")
    assert kv == 5 * WINDOW_FULL + FULL_LAYER
    assert label is True


def test_shared_kv_layers_own_no_cache():
    assert kv_layer_mask(_swa_meta(shared_kv_layers=2), 6) == (
        True,
        True,
        True,
        True,
        False,
        False,
    )
    kv, _ = _kv_of(_swa_meta(shared_kv_layers=2), {"ctx-size": 32768, "swa-full": True})
    assert kv == 4 * WINDOW_FULL


def test_draft_model_follows_the_window_rule():
    """The draft model's own window layers are priced by the same window
    rule as the main model, from its own head sizes and window token count.
    ctx-size-draft is not accepted for the llama.cpp engine in the catalog,
    so the draft falls back to the main ctx-size here."""
    ctx = 32768
    window = 1024
    ubatch = 512
    slots = 1
    d = _swa_meta(shared_kv_layers=0)
    est = estimate_memory(
        _swa_meta(),
        0,
        settings={
            "ctx-size": ctx,
            "parallel": slots,
            "ctx-size-draft": 16384,
            "n-gpu-layers": "all",
        },
        engine="llama.cpp",
        free_bytes_per_gpu=[1],
        draft_meta=d,
    )
    tokens = window_tokens(ctx, window, ubatch, slots, False)
    per_model_kv = 5 * (8 * 256 * 2 * 2 * tokens) + FULL_LAYER
    assert est.cards[0].kv == per_model_kv + per_model_kv


def test_vocab_term_is_charged_on_every_card(monkeypatch):
    """The vocabulary compute term adds f32 entries per token and vocabulary
    entry to the card formula, so it lands on every used card and not only
    on the output card."""
    monkeypatch.setitem(COMPUTE_TERMS, "vocab", 0.5)
    m = _meta(n_vocab=100000, n_layers=4, n_ff=256)
    est = estimate_memory(
        m,
        10**9,
        settings={"ctx-size": 2048, "tensor-split": "1,1"},
        engine="llama.cpp",
        free_bytes_per_gpu=[8 * 1024**3, 8 * 1024**3],
    )
    with_logits = compute_bytes(m, ubatch=512, ctx=2048, flash_attn=True, logits=True)
    non_output = next(i for i, c in enumerate(est.cards) if c.compute != with_logits)
    with_vocab = compute_bytes(
        m, ubatch=512, ctx=2048, flash_attn=True, logits=False, vocab=True
    )
    without_vocab = compute_bytes(
        m, ubatch=512, ctx=2048, flash_attn=True, logits=False, vocab=False
    )
    assert with_vocab != without_vocab
    assert est.cards[non_output].compute == with_vocab


def test_host_buffer_carries_no_vocabulary_activation(monkeypatch):
    """The host compute buffer excludes the vocabulary term, as it excludes
    the logits term: the CPU graph reserves neither."""
    monkeypatch.setitem(COMPUTE_TERMS, "vocab", 0.5)
    monkeypatch.setitem(COMPUTE_TERMS, "host", 1.0)
    m = _meta(n_vocab=1000, n_ff=0, n_embd=8)
    host = host_compute_bytes(m, ubatch=512, ctx=4096, flash_attn=True)
    assert host == int(4 * 512 * COMPUTE_TERMS["residual"] * 8)
