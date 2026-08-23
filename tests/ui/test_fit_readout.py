"""Configure-tab live fit readout: the model-meta label grows a live
"est vs free VRAM" line (node-aware, debounced, GPU probe off-thread) so the
user can tune ctx/KV-quant until the model fits BEFORE ever clicking Launch.
"""
import time

import llama_launcher.ui.main_window as mw
import llama_launcher.services.gpu as _gpu
from llama_launcher.core.gguf import GgufMeta
from llama_launcher.core.spec import Profile, Mount, Runtime
from llama_launcher.services.gpu import GpuStat


def _profile(ctx, **settings):
    return Profile(name="v", image="img", runtime=Runtime(binary="podman"),
                   mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
                   model="/models/m.gguf", settings={"port": 8080, "ctx-size": ctx, **settings})


def _gpu_stat(free_mib, total_mib=32768):
    return GpuStat(name="GPU", mem_used_mib=total_mib - free_mib, mem_total_mib=total_mib,
                   mem_free_mib=free_mib, util_pct=0, temp_c=40)


def _patch_model(monkeypatch, weights_gib=20):
    monkeypatch.setattr(mw.model_info, "read_gguf_meta",
        lambda path: GgufMeta(arch="llama", n_layers=80, n_head=64, n_head_kv=8,
                              n_embd=8192, ctx_train=131072, quant="Q8_0"))
    monkeypatch.setattr(mw.model_info, "file_size",
                        lambda path: weights_gib * 1024**3)


def _seed_gpus(panel, gpus, ssh=""):
    """Pre-warm the panel's GPU cache so _refresh_fit_line renders synchronously."""
    panel._fit_gpus = gpus
    panel._fit_gpus_ssh = ssh
    panel._fit_gpus_at = time.monotonic()


def test_fit_line_renders_when_fits(main_window, monkeypatch):
    _patch_model(monkeypatch)
    panel = main_window._configure_panel
    panel.load_profile(_profile(4096))
    _seed_gpus(panel, [_gpu_stat(30000)])
    panel._refresh_fit_line()
    t = panel.model_meta_label.text()
    assert "fit" in t and "free" in t


def test_fit_line_red_when_over_budget(main_window, monkeypatch):
    _patch_model(monkeypatch, weights_gib=40)
    panel = main_window._configure_panel
    panel.load_profile(_profile(131072))
    _seed_gpus(panel, [_gpu_stat(1024)])
    panel._refresh_fit_line()
    t = panel.model_meta_label.text()
    assert "may not fit" in t
    assert "color" in t          # over-budget renders styled/red


def test_fit_line_absent_without_gpus(main_window, monkeypatch):
    _patch_model(monkeypatch)
    panel = main_window._configure_panel
    panel.load_profile(_profile(4096))
    _seed_gpus(panel, [])
    panel._refresh_fit_line()
    assert "fit" not in panel.model_meta_label.text()


def test_fit_line_keeps_meta_text(main_window, monkeypatch):
    """The fit line joins the existing meta/caps text, never replaces it."""
    _patch_model(monkeypatch)
    panel = main_window._configure_panel
    panel.load_profile(_profile(4096))
    _seed_gpus(panel, [_gpu_stat(30000)])
    panel._refresh_fit_line()
    t = panel.model_meta_label.text()
    assert "Q8_0" in t           # quant from _meta_caps_text still present


def test_settings_change_schedules_debounced_refresh(main_window, monkeypatch):
    _patch_model(monkeypatch)
    panel = main_window._configure_panel
    panel.load_profile(_profile(4096))
    panel._fit_timer.stop()
    panel._widgets["ctx-size"].set_value(65536)   # fires changed -> refresh_preview
    assert panel._fit_timer.isActive()


def test_fit_gather_probes_profile_node(main_window, monkeypatch, qtbot):
    """A remote-node profile's readout probes THAT node's GPUs over ssh."""
    from llama_launcher.core.nodes import Node
    from llama_launcher.store.nodes import add_node
    add_node(Node(name="box-b", kind="remote", connection="box-b",
                  ssh_target="me@10.0.0.2"), main_window.base_dir())
    main_window._configure_panel.reload_nodes()
    _patch_model(monkeypatch)
    seen = {}

    def _query(ssh_target=""):
        seen["ssh"] = ssh_target
        return [_gpu_stat(30000)]
    monkeypatch.setattr(_gpu, "query_gpus", _query)
    panel = main_window._configure_panel
    p = _profile(4096)
    p.runtime.node = "box-b"
    panel.load_profile(p)
    panel._refresh_fit_line()    # cold cache -> dispatches the off-thread probe
    qtbot.waitUntil(lambda: "ssh" in seen, timeout=3000)
    assert seen["ssh"] == "me@10.0.0.2"
    qtbot.waitUntil(lambda: "fit" in panel.model_meta_label.text(), timeout=3000)
