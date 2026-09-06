"""Configure-tab live fit readout: the model-meta label carries a live
per-card and RAM fit breakdown (node-aware, debounced, GPU/RAM probe
off-thread) so the user can tune ctx/KV-quant until the model fits BEFORE
ever clicking Launch.
"""

import time

import llama_launcher.services.gpu as _gpu
import llama_launcher.ui.main_window as mw
from llama_launcher.core.gguf import GgufMeta, TensorInfo
from llama_launcher.core.spec import Mount, Profile, Runtime
from llama_launcher.services.gpu import GpuStat

_MIB = 1024 * 1024


def _profile(ctx, **settings):
    return Profile(
        name="v",
        image="img",
        runtime=Runtime(binary="podman"),
        mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
        model="/models/m.gguf",
        settings={"port": 8080, "ctx-size": ctx, **settings},
    )


def _gpu_stat(free_mib, total_mib=32768):
    return GpuStat(
        name="GPU",
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


def _patch_model(monkeypatch, weights_gib=20):
    """Patches read_model, keyed by host path, to return an 80-layer model
    for the profile's own model (/h/m.gguf), a genuinely smaller 2-layer
    model for the draft path (/h/d.gguf) and a small mmproj-sized file for
    /h/mm.gguf, so the draft and mmproj terms are non-zero and add less
    than a second copy of the main model would."""
    tensors = _layer_tensors(80)
    tensors.append(_tensor("token_embd.weight", 512 * _MIB))
    tensors.append(_tensor("output.weight", 512 * _MIB))
    meta = GgufMeta(
        arch="llama",
        n_layers=80,
        n_head=64,
        n_head_kv=8,
        n_embd=8192,
        ctx_train=131072,
        quant="Q8_0",
        tensors=tuple(tensors),
    )
    draft_meta = GgufMeta(
        arch="llama",
        n_layers=2,
        n_head=64,
        n_head_kv=8,
        n_embd=8192,
        ctx_train=131072,
        quant="Q8_0",
        tensors=tuple(_layer_tensors(2)),
    )

    def _read_model(path):
        p = str(path)
        if p == "/h/m.gguf":
            return meta, weights_gib * 1024**3
        if p == "/h/d.gguf":
            return draft_meta, 64 * _MIB
        if p == "/h/mm.gguf":
            return None, 64 * _MIB
        return None, None

    monkeypatch.setattr(mw.model_info, "read_model", _read_model)


def _seed_gpus(panel, gpus, ssh="", ram=64 * 1024**3):
    """Pre-warm the panel's GPU/RAM cache so _refresh_fit_line renders
    synchronously."""
    panel._fit_gpus = gpus
    panel._fit_gpus_ssh = ssh
    panel._fit_gpus_at = time.monotonic()
    panel._fit_ram = ram


def test_fit_lines_one_per_card_and_ram(main_window, monkeypatch):
    _patch_model(monkeypatch)
    panel = main_window._configure_panel
    panel.load_profile(_profile(4096, **{"tensor-split": "60,40"}))
    _seed_gpus(panel, [_gpu_stat(30000), _gpu_stat(20000)])
    panel._refresh_fit_line()
    t = panel.model_meta_label.text()
    assert "GPU0" in t and "GPU1" in t and "RAM" in t
    assert t.count("<br>") == 3
    assert "formula" in panel.model_meta_label.toolTip()


def test_fit_line_red_when_a_card_is_over(main_window, monkeypatch):
    _patch_model(monkeypatch)
    panel = main_window._configure_panel
    panel.load_profile(_profile(131072))
    _seed_gpus(panel, [_gpu_stat(1024)])
    panel._refresh_fit_line()
    t = panel.model_meta_label.text()
    assert "color" in t and "GPU0" in t


def test_label_wraps_instead_of_widening_the_window(main_window, monkeypatch):
    _patch_model(monkeypatch)
    panel = main_window._configure_panel
    assert panel.model_meta_label.wordWrap() is True
    panel.load_profile(_profile(4096, **{"tensor-split": "60,40"}))
    _seed_gpus(panel, [_gpu_stat(30000), _gpu_stat(20000)])
    panel._refresh_fit_line()
    assert panel.model_meta_label.minimumSizeHint().width() < 300


def test_ram_unknown_renders_without_verdict(main_window, monkeypatch):
    _patch_model(monkeypatch)
    panel = main_window._configure_panel
    panel.load_profile(_profile(4096))
    _seed_gpus(panel, [_gpu_stat(30000)], ram=None)
    panel._refresh_fit_line()
    assert "available unknown" in panel.model_meta_label.text()


def test_draft_and_mmproj_raise_the_estimate(main_window, monkeypatch):
    _patch_model(monkeypatch)
    panel = main_window._configure_panel
    p = _profile(4096)
    panel.load_profile(p)
    _seed_gpus(panel, [_gpu_stat(30000)])
    panel._refresh_fit_line()
    before = panel._current_fit_report().cards[0].est
    p.draft_model = "/models/d.gguf"
    p.mmproj = "/models/mm.gguf"
    panel.load_profile(p)
    panel._refresh_fit_line()
    after = panel._current_fit_report().cards[0].est
    assert after > before


def test_draft_widget_edit_raises_the_estimate_on_next_refresh(
    main_window, monkeypatch
):
    """Typing a draft-model path into the widget, without touching the loaded
    profile object, is picked up on the next refresh: the readout reads the
    draft path from current_profile(), which reads the widget live."""
    _patch_model(monkeypatch)
    panel = main_window._configure_panel
    panel.load_profile(_profile(4096))
    _seed_gpus(panel, [_gpu_stat(30000)])
    panel._refresh_fit_line()
    before = panel._current_fit_report().cards[0].est
    panel.draft_model_edit.setText("/models/d.gguf")
    panel._refresh_fit_line()
    after = panel._current_fit_report().cards[0].est
    assert after > before


def test_fit_line_absent_without_gpus(main_window, monkeypatch):
    _patch_model(monkeypatch)
    panel = main_window._configure_panel
    panel.load_profile(_profile(4096))
    _seed_gpus(panel, [])
    panel._refresh_fit_line()
    assert "GPU" not in panel.model_meta_label.text()


def test_fit_line_keeps_meta_text(main_window, monkeypatch):
    """The per-card readout joins the existing meta/caps text, never replaces it."""
    _patch_model(monkeypatch)
    panel = main_window._configure_panel
    panel.load_profile(_profile(4096))
    _seed_gpus(panel, [_gpu_stat(30000)])
    panel._refresh_fit_line()
    t = panel.model_meta_label.text()
    assert "Q8_0" in t  # quant from _meta_caps_text still present


def test_settings_change_schedules_debounced_refresh(main_window, monkeypatch):
    _patch_model(monkeypatch)
    panel = main_window._configure_panel
    panel.load_profile(_profile(4096))
    panel._fit_timer.stop()
    panel._widgets["ctx-size"].set_value(65536)  # fires changed -> refresh_preview
    assert panel._fit_timer.isActive()


def test_fit_gather_probes_profile_node(main_window, monkeypatch, qtbot):
    """A remote-node profile's readout probes THAT node's GPUs over ssh."""
    from llama_launcher.core.nodes import Node
    from llama_launcher.store.nodes import add_node

    add_node(
        Node(name="box-b", kind="remote", connection="box-b", ssh_target="me@10.0.0.2"),
        main_window.base_dir(),
    )
    main_window._configure_panel.reload_nodes()
    _patch_model(monkeypatch)
    seen = {}

    def _query(ssh_target=""):
        seen["ssh"] = ssh_target
        return [_gpu_stat(30000)]

    from llama_launcher.services import pool_preflight

    monkeypatch.setattr(_gpu, "query_gpus", _query)
    monkeypatch.setattr(pool_preflight, "free_ram_bytes", lambda ssh="": 0)
    panel = main_window._configure_panel
    p = _profile(4096)
    p.runtime.node = "box-b"
    panel.load_profile(p)
    panel._refresh_fit_line()  # cold cache -> dispatches the off-thread probe
    qtbot.waitUntil(lambda: "ssh" in seen, timeout=3000)
    assert seen["ssh"] == "me@10.0.0.2"
    qtbot.waitUntil(lambda: "GPU0" in panel.model_meta_label.text(), timeout=3000)


# -- router mode: the fit line comes from the MEMBERS, not the stale form model


def _patch_models_by_path(monkeypatch, sizes_gib_by_host):
    meta = GgufMeta(
        arch="llama",
        n_layers=80,
        n_head=64,
        n_head_kv=8,
        n_embd=8192,
        ctx_train=131072,
        quant="Q8_0",
    )

    def _read_model(path):
        size = int(sizes_gib_by_host.get(str(path), 1) * 1024**3)
        return meta, size

    monkeypatch.setattr(mw.model_info, "read_model", _read_model)


def _router_with_member(main_window, member_gib, monkeypatch, stale_gib=40):
    """Load a profile whose own model is huge, switch to router mode, add a
    small saved member. The estimate must track the member, not the leftover."""
    from llama_launcher.core.spec import RouterMember
    from llama_launcher.store import profiles as store

    _patch_models_by_path(
        monkeypatch, {"/h/m.gguf": stale_gib, "/mnt/models/small.gguf": member_gib}
    )
    store.save_profile(
        Profile(
            name="Small",
            image="img",
            model="/models/small.gguf",
            mounts=[Mount(host="/mnt/models", container="/models", role="model")],
            settings={"ctx-size": 4096},
        ),
        main_window.router_base_dir(),
    )
    panel = main_window._configure_panel
    panel.load_profile(_profile(4096))  # form still holds /models/m.gguf
    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("router"))
    panel._add_member_item(RouterMember(profile="Small"))
    return panel


def test_router_fit_uses_member_model_not_stale_form_model(main_window, monkeypatch):
    panel = _router_with_member(main_window, member_gib=4, monkeypatch=monkeypatch)
    _seed_gpus(panel, [_gpu_stat(30000)])  # 4 GiB member fits; 40 GiB stale would not
    panel._refresh_fit_line()
    t = panel.model_meta_label.text()
    assert "fit" in t
    assert "may not fit" not in t


def test_router_fit_warns_when_member_too_big(main_window, monkeypatch):
    panel = _router_with_member(main_window, member_gib=40, monkeypatch=monkeypatch)
    _seed_gpus(panel, [_gpu_stat(30000)])
    panel._refresh_fit_line()
    assert "may not fit" in panel.model_meta_label.text()


def test_router_fit_hides_stale_model_meta(main_window, monkeypatch):
    """A router serves its members' models; the leftover form model's meta/caps
    text must not linger next to the member-based fit line."""
    panel = _router_with_member(main_window, member_gib=4, monkeypatch=monkeypatch)
    _seed_gpus(panel, [_gpu_stat(30000)])
    panel._refresh_fit_line()
    assert "Q8_0" not in panel.model_meta_label.text()


def test_switching_to_router_clears_the_single_server_tooltip(main_window, monkeypatch):
    """The per-card breakdown tooltip belongs to the single-server readout;
    a router render must not keep it lingering on the label."""
    from llama_launcher.core.spec import RouterMember
    from llama_launcher.store import profiles as store

    _patch_model(monkeypatch)
    panel = main_window._configure_panel
    panel.load_profile(_profile(4096))
    _seed_gpus(panel, [_gpu_stat(30000)])
    panel._refresh_fit_line()
    assert panel.model_meta_label.toolTip() != ""

    _patch_models_by_path(monkeypatch, {"/mnt/models/small.gguf": 4})
    store.save_profile(
        Profile(
            name="Small",
            image="img",
            model="/models/small.gguf",
            mounts=[Mount(host="/mnt/models", container="/models", role="model")],
            settings={"ctx-size": 4096},
        ),
        main_window.router_base_dir(),
    )
    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("router"))
    panel._add_member_item(RouterMember(profile="Small"))
    panel._refresh_fit_line()
    assert panel.model_meta_label.toolTip() == ""


def test_member_with_unreadable_header_counts_at_its_file_size(
    main_window, monkeypatch
):
    """A router member whose header cannot be parsed (meta None) still
    counts toward the pool estimate at its file size, rather than being
    dropped from the sum entirely."""
    from llama_launcher.core.spec import RouterMember
    from llama_launcher.store import profiles as store

    meta = GgufMeta(
        arch="llama",
        n_layers=80,
        n_head=64,
        n_head_kv=8,
        n_embd=8192,
        ctx_train=131072,
        quant="Q8_0",
    )

    def _read_model(path):
        if str(path) == "/mnt/models/broken.gguf":
            return None, 40 * 1024**3
        return meta, 4 * 1024**3

    monkeypatch.setattr(mw.model_info, "read_model", _read_model)
    store.save_profile(
        Profile(
            name="Small",
            image="img",
            model="/models/small.gguf",
            mounts=[Mount(host="/mnt/models", container="/models", role="model")],
            settings={"ctx-size": 4096},
        ),
        main_window.router_base_dir(),
    )
    store.save_profile(
        Profile(
            name="Broken",
            image="img",
            model="/models/broken.gguf",
            mounts=[Mount(host="/mnt/models", container="/models", role="model")],
            settings={"ctx-size": 4096},
        ),
        main_window.router_base_dir(),
    )
    panel = main_window._configure_panel
    panel.load_profile(_profile(4096))
    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("router"))
    panel._add_member_item(RouterMember(profile="Small"))
    without_broken = sum(panel._member_estimates())
    panel._add_member_item(RouterMember(profile="Broken"))
    with_broken = sum(panel._member_estimates())
    assert with_broken > without_broken


def test_cached_meta_weights_invalidates_when_a_split_sibling_changes(
    main_window, tmp_path, monkeypatch
):
    """The cache stamp covers every part of a split model, named from the
    file's own part-of-total suffix: replacing part 2 on disk is picked up
    on the next read even though part 1 itself is untouched."""
    from llama_launcher.services import model_info

    p1 = tmp_path / "m-00001-of-00002.gguf"
    p2 = tmp_path / "m-00002-of-00002.gguf"
    p1.write_bytes(b"a" * 10)
    p2.write_bytes(b"b" * 20)
    metas = {
        str(p1): GgufMeta(
            arch="llama",
            n_layers=2,
            split_count=2,
            tensors=(TensorInfo("blk.0.attn_q.weight", 1, 0, 4),),
        ),
        str(p2): GgufMeta(
            arch="llama",
            n_layers=2,
            split_count=2,
            tensors=(TensorInfo("blk.1.attn_q.weight", 1, 0, 4),),
        ),
    }
    monkeypatch.setattr(
        model_info, "read_gguf_meta", lambda path, **kw: metas.get(str(path))
    )
    panel = main_window._configure_panel
    mounts = [Mount(host=str(tmp_path), container="/models", role="model")]
    _meta1, weights1 = panel._cached_meta_weights(
        "/models/m-00001-of-00002.gguf", mounts
    )
    assert weights1 == 30

    p2.write_bytes(b"b" * 50)  # replace part 2 with a bigger file
    _meta2, weights2 = panel._cached_meta_weights(
        "/models/m-00001-of-00002.gguf", mounts
    )
    assert weights2 == 60
