import llama_launcher.ui.main_window as mw
from llama_launcher.core.gguf import GgufMeta
from llama_launcher.core.spec import Mount, Profile, Runtime
from llama_launcher.services.gpu import GpuStat


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


def test_vram_check_warns_when_over(qtbot, monkeypatch):
    monkeypatch.setattr(
        mw.model_info,
        "read_gguf_meta",
        lambda path: GgufMeta(
            arch="llama",
            n_layers=80,
            n_head=64,
            n_head_kv=8,
            n_embd=8192,
            ctx_train=131072,
            quant="Q8_0",
        ),
    )
    monkeypatch.setattr(mw.model_info, "file_size", lambda path: 20 * 1024**3)
    monkeypatch.setattr(
        mw.gpu, "query_gpus", lambda ssh_target="": [_gpu(1024)]
    )  # ~1 GiB free
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(_profile(131072))
    msg = w._launch.vram_check()
    assert msg is not None and "VRAM" in msg
    assert "--n-cpu-moe" in msg and "--n-cpu-ffn" in msg


def test_vram_check_none_when_unknown(qtbot, monkeypatch):
    monkeypatch.setattr(mw.model_info, "read_gguf_meta", lambda path: None)
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda ssh_target="": [])
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(_profile(4096))
    assert w._launch.vram_check() is None


def test_vram_check_sums_free_across_two_gpus(qtbot, monkeypatch):
    # 16+12 GB rig, 14.7 + 7.3 GiB free, model ~20.2 GiB. Split across both
    # GPUs it fits (~22 GiB), so NO warning: the budget is the sum across
    # GPUs, not one card's max.
    gib = 1024**3
    monkeypatch.setattr(
        mw.model_info,
        "read_gguf_meta",
        lambda path: GgufMeta(
            arch="llama",
            n_layers=1,
            n_head=8,
            n_head_kv=8,
            n_embd=64,
            ctx_train=4096,
            quant="Q8_0",
        ),
    )
    monkeypatch.setattr(mw.model_info, "file_size", lambda path: int(20.0 * gib))
    monkeypatch.setattr(
        mw.gpu,
        "query_gpus",
        lambda ssh_target="": [
            _gpu(int(14.7 * 1024)),
            _gpu(int(7.3 * 1024), total_mib=12288),
        ],
    )
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(_profile(4096))  # default split-mode -> summed
    assert w._launch.vram_check() is None  # ~22 GiB free covers ~20 GiB


def test_vram_check_split_none_uses_single_gpu(qtbot, monkeypatch):
    # split-mode none puts everything on main-gpu, so the same model that fits
    # across both GPUs does not fit on one 16 GB card -> warns, and the message
    # reports the single-card free, not the sum.
    gib = 1024**3
    monkeypatch.setattr(
        mw.model_info,
        "read_gguf_meta",
        lambda path: GgufMeta(
            arch="llama",
            n_layers=1,
            n_head=8,
            n_head_kv=8,
            n_embd=64,
            ctx_train=4096,
            quant="Q8_0",
        ),
    )
    monkeypatch.setattr(mw.model_info, "file_size", lambda path: int(20.0 * gib))
    monkeypatch.setattr(
        mw.gpu,
        "query_gpus",
        lambda ssh_target="": [
            _gpu(int(14.7 * 1024)),
            _gpu(int(7.3 * 1024), total_mib=12288),
        ],
    )
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(_profile(4096, **{"split-mode": "none"}))
    msg = w._launch.vram_check()
    assert msg is not None
    assert "across" not in msg  # single-GPU budget, no breakdown


def test_vram_check_shows_per_gpu_breakdown(qtbot, monkeypatch):
    # When it genuinely doesn't fit across multiple GPUs, the message shows the
    # per-card free so the "free" figure is transparent.
    gib = 1024**3
    monkeypatch.setattr(
        mw.model_info,
        "read_gguf_meta",
        lambda path: GgufMeta(
            arch="llama",
            n_layers=80,
            n_head=64,
            n_head_kv=8,
            n_embd=8192,
            ctx_train=131072,
            quant="Q8_0",
        ),
    )
    monkeypatch.setattr(mw.model_info, "file_size", lambda path: 40 * gib)
    monkeypatch.setattr(
        mw.gpu,
        "query_gpus",
        lambda ssh_target="": [
            _gpu(int(14.7 * 1024)),
            _gpu(int(7.3 * 1024), total_mib=12288),
        ],
    )
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(_profile(131072))
    msg = w._launch.vram_check()
    assert msg is not None and "across 2 GPUs" in msg and "+" in msg


def test_vram_check_uses_profile_nodes_gpus(main_window, monkeypatch):
    """A profile pinned to a remote node must be judged against THAT node's
    free VRAM (ssh nvidia-smi), not the local cards."""
    from llama_launcher.core.nodes import Node
    from llama_launcher.core.spec import Mount, Profile, Runtime
    from llama_launcher.store.nodes import add_node

    add_node(
        Node(name="box-b", kind="remote", connection="box-b", ssh_target="me@10.0.0.2"),
        main_window.base_dir(),
    )
    main_window._configure_panel.reload_nodes()  # combo predates the add
    monkeypatch.setattr(
        mw.model_info,
        "read_gguf_meta",
        lambda path: GgufMeta(
            arch="llama",
            n_layers=80,
            n_head=64,
            n_head_kv=8,
            n_embd=8192,
            ctx_train=131072,
            quant="Q8_0",
        ),
    )
    monkeypatch.setattr(mw.model_info, "file_size", lambda path: 20 * 1024**3)
    seen = {}

    def _query(ssh_target=""):
        seen["ssh"] = ssh_target
        return [_gpu(1024)]

    monkeypatch.setattr(mw.gpu, "query_gpus", _query)
    p = Profile(
        name="v",
        image="img",
        runtime=Runtime(binary="podman", node="box-b"),
        mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
        model="/models/m.gguf",
        settings={"port": 8080, "ctx-size": 131072},
    )
    main_window._configure_panel.load_profile(p)
    assert main_window._launch.vram_check() is not None
    assert seen["ssh"] == "me@10.0.0.2"


def test_check_fit_estimate_honours_per_slot_context_and_engine(qtbot, monkeypatch):
    # _model_estimate_bytes backs the pooled "Check fit" readout; it must
    # follow the same effective-context rule as the launch-time preflight
    # (parallel * kv-unified-per-slot on an engine that accepts the flag,
    # the model's trained context otherwise) rather than reading ctx-size raw.
    monkeypatch.setattr(
        mw.model_info,
        "read_gguf_meta",
        lambda path: GgufMeta(
            arch="llama",
            n_layers=2,
            n_head=8,
            n_head_kv=4,
            n_embd=64,
            ctx_train=4096,
            quant="Q8_0",
        ),
    )
    monkeypatch.setattr(mw.model_info, "file_size", lambda path: 1000)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    panel = w._configure_panel

    explicit = _profile(16384)
    per_slot = _profile(None, **{"kv-unified-per-slot": 4096, "parallel": 4})
    on_ik = Profile(
        name="v",
        image="img",
        runtime=Runtime(binary="podman", engine="ik_llama.cpp"),
        mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
        model="/models/m.gguf",
        settings={"port": 8080, "kv-unified-per-slot": 4096, "parallel": 4},
    )
    trained_ctx = _profile(None)

    panel.load_profile(explicit)
    explicit_bytes = panel._model_estimate_bytes(explicit)
    panel.load_profile(per_slot)
    per_slot_bytes = panel._model_estimate_bytes(per_slot)
    panel.load_profile(on_ik)
    ik_bytes = panel._model_estimate_bytes(on_ik)
    panel.load_profile(trained_ctx)
    trained_bytes = panel._model_estimate_bytes(trained_ctx)

    assert per_slot_bytes == explicit_bytes
    assert ik_bytes == trained_bytes
    assert per_slot_bytes != ik_bytes
