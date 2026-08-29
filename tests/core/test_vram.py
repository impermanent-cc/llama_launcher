from types import SimpleNamespace

from llama_launcher.core.vram import (
    bytes_per_elem, kv_cache_bytes, estimate, fits, VramEstimate, pooled_fit,
    estimate_for_model,
)


def _meta(**kw):
    base = dict(n_layers=2, n_head=8, n_head_kv=4, n_embd=64, ctx_train=4096)
    base.update(kw)
    return SimpleNamespace(**base)


def test_bytes_per_elem():
    assert bytes_per_elem("f16") == 2.0
    assert bytes_per_elem("q8_0") == 1.0625
    assert bytes_per_elem("unknown") == 2.0   # safe default


def test_kv_cache_bytes_f16():
    # 2 layers, 4 kv-heads, head_dim 8, ctx 100, f16 (2 bytes), K and V
    per = 2 * 100 * 4 * 8
    assert kv_cache_bytes(2, 4, 8, 100, "f16", "f16") == int(per * 2 + per * 2)


def test_estimate_total():
    e = estimate(n_layers=2, n_head=8, n_head_kv=4, n_embd=64, ctx=100,
                 weights_bytes=1000, overhead_bytes=500)
    assert isinstance(e, VramEstimate)
    assert e.total_bytes == e.kv_bytes + 1000 + 500


def test_fits():
    assert fits(100, 200) == (True, 100)
    ok, margin = fits(300, 200)
    assert ok is False and margin == -100


def test_fits_boundary_zero_margin():
    """When estimate equals free exactly, margin==0 and fits==True."""
    assert fits(200, 200) == (True, 0)


def test_available_free_bytes_sums_across_gpus_when_split():
    from llama_launcher.core.vram import available_free_bytes
    # Regression: a model split across GPUs (default split-mode) should see the
    # COMBINED free VRAM, not just the largest single card. 16+12 GB rig with
    # 14.7 + 7.3 GiB free must offer ~22 GiB, so a 20.2 GiB model fits.
    gib = 1024 ** 3
    free = [int(14.7 * gib), int(7.3 * gib)]
    assert available_free_bytes(free, "layer", 0) == sum(free)
    assert available_free_bytes(free, "row", 0) == sum(free)
    assert available_free_bytes(free, "tensor", 0) == sum(free)


def test_available_free_bytes_uses_main_gpu_when_split_none():
    from llama_launcher.core.vram import available_free_bytes
    gib = 1024 ** 3
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
    gb = 1024 ** 3
    r = pooled_fit(120 * gb, [("vram", 48 * gb), ("ram", 96 * gb)])
    assert r.fits is True
    assert r.vram_bytes == 48 * gb and r.ram_bytes == 96 * gb
    assert r.total_bytes == 144 * gb and r.margin == 24 * gb


def test_pooled_fit_does_not_fit_is_negative_margin():
    gb = 1024 ** 3
    r = pooled_fit(200 * gb, [("vram", 48 * gb), ("ram", 96 * gb)])
    assert r.fits is False and r.margin == -56 * gb


def test_estimate_for_model_matches_estimate_total_bytes():
    """The shared weights+KV estimate used by both the single-node (vram_check)
    and pooled (Check fit) preflights equals a direct estimate().total_bytes."""
    meta = _meta()
    got = estimate_for_model(meta, 1000, ctx_size=100,
                             k_quant="f16", v_quant="f16")
    direct = estimate(n_layers=2, n_head=8, n_head_kv=4, n_embd=64, ctx=100,
                      k_quant="f16", v_quant="f16", weights_bytes=1000)
    assert got == direct.total_bytes


def test_estimate_for_model_falls_back_to_ctx_train_then_default():
    # ctx_size None -> meta.ctx_train
    on_train = estimate_for_model(_meta(ctx_train=512), 0, ctx_size=None)
    direct_train = estimate(n_layers=2, n_head=8, n_head_kv=4, n_embd=64,
                            ctx=512, weights_bytes=0)
    assert on_train == direct_train.total_bytes
    # ctx_size None and ctx_train falsy -> 4096
    on_default = estimate_for_model(_meta(ctx_train=0), 0, ctx_size=None)
    direct_default = estimate(n_layers=2, n_head=8, n_head_kv=4, n_embd=64,
                              ctx=4096, weights_bytes=0)
    assert on_default == direct_default.total_bytes


def test_estimate_for_model_returns_weights_when_meta_insufficient():
    assert estimate_for_model(None, 1234) == 1234
    assert estimate_for_model(_meta(n_layers=0), 1234) == 1234
    assert estimate_for_model(_meta(n_embd=0), 1234) == 1234
    assert estimate_for_model(None, None) == 0


# -- fit_summary: the one shared estimate-vs-free computation ----------------

def test_fit_summary_fits_with_margin():
    from llama_launcher.core.vram import fit_summary
    est = estimate_for_model(_meta(), 1000, ctx_size=100)
    s = fit_summary(_meta(), 1000, settings={"ctx-size": 100},
                    free_bytes_per_gpu=[est + 5, est + 5])
    assert s.fits and s.margin == est + 10 and s.est_bytes == est
    assert s.free_bytes == 2 * est + 10


def test_fit_summary_over_budget_negative_margin():
    from llama_launcher.core.vram import fit_summary
    est = estimate_for_model(_meta(), 10_000, ctx_size=100)
    s = fit_summary(_meta(), 10_000, settings={"ctx-size": 100},
                    free_bytes_per_gpu=[est - 7])
    assert not s.fits and s.margin == -7


def test_fit_summary_split_none_uses_main_gpu_only():
    from llama_launcher.core.vram import fit_summary
    est = estimate_for_model(_meta(), 0, ctx_size=100)
    s = fit_summary(_meta(), 0, settings={"ctx-size": 100, "split-mode": "none",
                                          "main-gpu": 1},
                    free_bytes_per_gpu=[0, est + 3])
    assert s.fits and s.free_bytes == est + 3


def test_fit_summary_honors_kv_quant():
    from llama_launcher.core.vram import fit_summary
    f16 = fit_summary(_meta(), 0, settings={"ctx-size": 4096},
                      free_bytes_per_gpu=[10**12])
    q8 = fit_summary(_meta(), 0, settings={"ctx-size": 4096,
                                           "cache-type-k": "q8_0",
                                           "cache-type-v": "q8_0"},
                     free_bytes_per_gpu=[10**12])
    assert q8.est_bytes < f16.est_bytes


def test_fit_summary_none_when_unknowable():
    from llama_launcher.core.vram import fit_summary
    assert fit_summary(None, 1000, settings={}, free_bytes_per_gpu=[10**9]) is None
    assert fit_summary(_meta(n_layers=0), 1000, settings={},
                       free_bytes_per_gpu=[10**9]) is None
    assert fit_summary(_meta(), 1000, settings={}, free_bytes_per_gpu=[]) is None


def test_router_fit_sums_k_largest():
    from llama_launcher.core.vram import router_fit_summary
    s = router_fit_summary([10, 30, 20], models_max=2, free_bytes_per_gpu=[60])
    assert s.est_bytes == 50            # 30 + 20, the two largest
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
