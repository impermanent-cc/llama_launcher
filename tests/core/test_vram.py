from llama_launcher.core.vram import (
    bytes_per_elem, kv_cache_bytes, estimate, fits, VramEstimate,
)


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
