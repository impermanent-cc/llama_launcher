"""Launch-time VRAM/RAM preflight: LaunchController.vram_check() builds the
same per-card and RAM fit report the Configure tab's live readout uses and
renders it as the abortable launch dialog's text.
"""

import pytest

import llama_launcher.ui.main_window as mw
from llama_launcher.core.gguf import GgufMeta, TensorInfo
from llama_launcher.core.spec import Mount, Profile, Runtime
from llama_launcher.services import pool_preflight
from llama_launcher.services.gpu import GpuStat

_MIB = 1024 * 1024


@pytest.fixture(autouse=True)
def _fixed_free_ram(monkeypatch):
    """Every preflight test judges RAM against a fixed figure instead of
    the host's real /proc/meminfo (or an ssh round trip), so a test's own
    override, set after this fixture runs, still wins for that test."""
    monkeypatch.setattr(
        pool_preflight, "free_ram_bytes", lambda ssh_target="": 64 * 1024**3
    )


def _profile(ctx, **settings):
    return Profile(
        name="v",
        image="img",
        runtime=Runtime(binary="podman"),
        mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
        model="/models/m.gguf",
        settings={"port": 8080, "ctx-size": ctx, **settings},
    )


def _gpu(free_mib, total_mib=16384, name="GPU"):
    return GpuStat(
        name=name,
        mem_used_mib=total_mib - free_mib,
        mem_total_mib=total_mib,
        mem_free_mib=free_mib,
        util_pct=0,
        temp_c=40,
    )


def _tensor(name: str, nbytes: int) -> TensorInfo:
    """A type-0 (F32) tensor of exactly nbytes bytes."""
    return TensorInfo(name=name, n_elements=nbytes // 4, ggml_type=0, nbytes=nbytes)


def _layer_tensors(n_layers: int) -> list:
    tensors = []
    for i in range(n_layers):
        tensors.append(_tensor(f"blk.{i}.attn_q.weight", 64 * _MIB))
        tensors.append(_tensor(f"blk.{i}.ffn_up.weight", 192 * _MIB))
    return tensors


def _meta(n_layers: int = 80) -> GgufMeta:
    tensors = _layer_tensors(n_layers)
    tensors.append(_tensor("token_embd.weight", 512 * _MIB))
    tensors.append(_tensor("output.weight", 512 * _MIB))
    return GgufMeta(
        arch="llama",
        n_layers=n_layers,
        n_head=64,
        n_head_kv=8,
        n_embd=8192,
        ctx_train=131072,
        quant="Q8_0",
        tensors=tuple(tensors),
    )


def _patch_model(monkeypatch, weights_gib=20, n_layers=80):
    """Patches read_model to return the profile's own model (host /h/m.gguf)
    with a full tensor table, and (None, None) for any other host path, so a
    test that sets a draft or projector path needs its own read_model stub."""
    meta = _meta(n_layers)

    def _read_model(path):
        if str(path) == "/h/m.gguf":
            return meta, weights_gib * 1024**3
        return None, None

    monkeypatch.setattr(mw.model_info, "read_model", _read_model)


def test_vram_check_warns_when_over(main_window, monkeypatch):
    # A small model whose weights, once fully on GPU, exceed a tight budget,
    # but which fits once every FFN layer's weights offload to the CPU: the
    # dialog names the card, the shortfall and the offload that fixes it.
    _patch_model(monkeypatch, n_layers=4)
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda ssh_target="": [_gpu(2048)])
    main_window._configure_panel.load_profile(_profile(4096))
    text = main_window._launch.vram_check()
    assert "GPU0" in text and "exceeds" in text
    assert "--n-cpu-ffn" in text


def test_vram_check_includes_draft_weights(main_window, monkeypatch):
    """The draft model's own weights, KV and compute add to the estimate: a
    profile that fits without a draft can overflow once one is set."""
    meta = _meta(n_layers=4)
    draft_meta = _meta(n_layers=2)

    def _read_model(path):
        if str(path) == "/h/m.gguf":
            return meta, 0
        if str(path) == "/h/d.gguf":
            return draft_meta, 0
        return None, None

    monkeypatch.setattr(mw.model_info, "read_model", _read_model)
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda ssh_target="": [_gpu(3400)])
    p = _profile(4096)
    main_window._configure_panel.load_profile(p)
    assert main_window._launch.vram_check() is None

    p.draft_model = "/models/d.gguf"
    main_window._configure_panel.load_profile(p)
    text = main_window._launch.vram_check()
    assert text is not None and "GPU0" in text


def test_fit_report_for_forwards_raw_args_and_projector(main_window, monkeypatch):
    """raw_args reaches the estimate the same as settings, and the
    projector's bytes land on the GPU: a raw -ngl 0 pushes the model's
    weights to RAM, leaving only the projector on the card, so the card
    estimate drops and the RAM figure rises."""
    meta = _meta(n_layers=4)

    def _read_model(path):
        if str(path) == "/h/m.gguf":
            return meta, 0
        if str(path) == "/h/mm.gguf":
            return None, 64 * _MIB
        return None, None

    monkeypatch.setattr(mw.model_info, "read_model", _read_model)
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda ssh_target="": [_gpu(30000)])
    p = _profile(4096)
    p.mmproj = "/models/mm.gguf"
    ctl = main_window._launch

    plain = ctl._fit_report_for(p)
    p.raw_args = "-ngl 0"
    limited = ctl._fit_report_for(p)

    assert limited.cards[0].est < plain.cards[0].est
    assert limited.ram.est > plain.ram.est


def test_vram_check_none_without_any_gpu(main_window, monkeypatch):
    _patch_model(monkeypatch)
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda ssh_target="": [])
    main_window._configure_panel.load_profile(_profile(131072))
    assert main_window._launch.vram_check() is None


def test_vram_check_none_when_fits(main_window, monkeypatch):
    _patch_model(monkeypatch, weights_gib=1)
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda ssh_target="": [_gpu(30000)])
    main_window._configure_panel.load_profile(_profile(4096))
    assert main_window._launch.vram_check() is None


def test_vram_check_fit_unset_mentions_shrink(main_window, monkeypatch):
    _patch_model(monkeypatch)
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda ssh_target="": [_gpu(1024)])
    # No fit, n-gpu-layers, override-tensor or ctx-size set: llama.cpp's
    # --fit runs by default and, with no context pinned, an over-budget
    # profile is told what --fit will shrink the context to instead of a
    # flat "won't fit".
    main_window._configure_panel.load_profile(_profile(131072, **{"ctx-size": None}))
    text = main_window._launch.vram_check()
    assert "shrink" in text


def test_vram_check_fit_on_explicit_is_silent(main_window, monkeypatch):
    _patch_model(monkeypatch)
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda ssh_target="": [_gpu(1024)])
    main_window._configure_panel.load_profile(_profile(131072, fit="on"))
    assert main_window._launch.vram_check() is None


def test_vram_check_two_cards_names_the_overflowing_card(main_window, monkeypatch):
    _patch_model(monkeypatch)
    monkeypatch.setattr(
        mw.gpu,
        "query_gpus",
        lambda ssh_target="": [_gpu(30000), _gpu(1024, total_mib=12288)],
    )
    main_window._configure_panel.load_profile(
        _profile(4096, **{"tensor-split": "50,50"})
    )
    text = main_window._launch.vram_check()
    # The dialog leads with the full per-card breakdown, so GPU0 appears
    # there too; only the overflowing card's shortfall message, after that
    # breakdown block, names it as exceeding free VRAM.
    message_part = text.split("\n\n", 1)[1]
    assert "GPU1: est" in message_part
    assert "GPU0: est" not in message_part


def test_vram_check_ram_warning(main_window, monkeypatch):
    _patch_model(monkeypatch)
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda ssh_target="": [_gpu(30000)])
    monkeypatch.setattr(
        pool_preflight, "free_ram_bytes", lambda ssh_target="": 256 * _MIB
    )
    main_window._configure_panel.load_profile(_profile(4096, **{"n-cpu-ffn": 80}))
    text = main_window._launch.vram_check()
    assert text is not None and "RAM" in text


def test_vram_check_uses_profile_nodes_gpus(main_window, monkeypatch):
    """A profile pinned to a remote node must be judged against THAT node's
    free VRAM (ssh nvidia-smi), not the local cards."""
    from llama_launcher.core.nodes import Node
    from llama_launcher.store.nodes import add_node

    add_node(
        Node(name="box-b", kind="remote", connection="box-b", ssh_target="me@10.0.0.2"),
        main_window.base_dir(),
    )
    main_window._configure_panel.reload_nodes()  # combo predates the add
    _patch_model(monkeypatch)
    seen = {}

    def _query(ssh_target=""):
        seen["ssh"] = ssh_target
        return [_gpu(1024)]

    monkeypatch.setattr(mw.gpu, "query_gpus", _query)
    p = _profile(131072)
    p.runtime.node = "box-b"
    main_window._configure_panel.load_profile(p)
    assert main_window._launch.vram_check() is not None
    assert seen["ssh"] == "me@10.0.0.2"


def test_fit_report_for_rereads_the_main_model_every_call(main_window, monkeypatch):
    """The launch preflight always reads the main model through the panel's
    stat-cached reader, one stat per click, rather than trusting the form's
    last render: a file replaced after the profile loaded is picked up on
    the next preflight."""
    small = _meta(n_layers=2)
    big = _meta(n_layers=8)

    def _read_small(path):
        if str(path) == "/h/m.gguf":
            return small, 0
        return None, None

    monkeypatch.setattr(mw.model_info, "read_model", _read_small)
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda ssh_target="": [_gpu(30000)])
    p = _profile(4096)
    main_window._configure_panel.load_profile(p)
    ctl = main_window._launch
    before = ctl._fit_report_for(p)

    def _read_big(path):
        if str(path) == "/h/m.gguf":
            return big, 0
        return None, None

    monkeypatch.setattr(mw.model_info, "read_model", _read_big)
    after = ctl._fit_report_for(p)
    assert after.cards[0].est > before.cards[0].est
