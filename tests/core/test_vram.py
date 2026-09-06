from types import SimpleNamespace

from llama_launcher.core.gguf import GgufMeta, TensorInfo
from llama_launcher.core.vram import (
    CARD_OVERHEAD_BYTES,
    COMPUTE_TERMS,
    bytes_per_elem,
    compute_bytes,
    effective_ctx_size,
    estimate_memory,
    fits,
    kv_cache_bytes,
    pooled_fit,
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
    base = dict(n_layers=2, n_head=8, n_head_kv=4, n_embd=64, ctx_train=4096)
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
    assert e.ram.buffers == 1000 * 1024 * 4
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
