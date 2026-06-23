import llama_launcher.ui.main_window as mw
from llama_launcher.core.spec import Profile, Mount, Runtime
from llama_launcher.core.gguf import GgufMeta


def _profile(ctx):
    return Profile(name="v", image="img", runtime=Runtime(binary="podman"),
                   mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
                   model="/models/m.gguf", settings={"port": 8080, "ctx-size": ctx})


def test_vram_check_warns_when_over(qtbot, monkeypatch):
    monkeypatch.setattr(mw.model_info, "read_gguf_meta",
        lambda path: GgufMeta(arch="llama", n_layers=80, n_head=64, n_head_kv=8,
                              n_embd=8192, ctx_train=131072, quant="Q8_0"))
    monkeypatch.setattr(mw.model_info, "file_size", lambda path: 20 * 1024**3)
    monkeypatch.setattr(mw.gpu, "free_vram_bytes", lambda: 1 * 1024**3)  # tiny
    w = mw.MainWindow(); qtbot.addWidget(w)
    w.load_profile(_profile(131072))
    msg = w.vram_check()
    assert msg is not None and "VRAM" in msg


def test_vram_check_none_when_unknown(qtbot, monkeypatch):
    monkeypatch.setattr(mw.model_info, "read_gguf_meta", lambda path: None)
    monkeypatch.setattr(mw.gpu, "free_vram_bytes", lambda: None)
    w = mw.MainWindow(); qtbot.addWidget(w)
    w.load_profile(_profile(4096))
    assert w.vram_check() is None
