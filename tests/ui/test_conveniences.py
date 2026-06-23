import os
import llama_launcher.ui.main_window as mw
from llama_launcher.core.spec import Profile, Mount, Runtime, LoraRef
from llama_launcher.core.gguf import GgufMeta
from llama_launcher.ui.panels.lora_panel import LoraPanel


def _profile():
    return Profile(name="c", image="img", runtime=Runtime(binary="podman"),
                   mounts=[Mount(host="/mnt/Models", container="/models", role="model", mode="ro")],
                   model="/models/m.gguf", settings={"port": 8080})


def test_open_web_ui_invokes_xdg(qtbot, monkeypatch):
    captured = {}
    monkeypatch.setattr(mw.subprocess, "Popen", lambda argv, **k: captured.setdefault("argv", argv))
    w = mw.MainWindow(); qtbot.addWidget(w)
    w.load_profile(_profile())
    w.open_web_ui()
    assert captured["argv"][0] == "xdg-open"
    assert captured["argv"][1] == "http://127.0.0.1:8080"


def test_export_sh(qtbot, tmp_path):
    w = mw.MainWindow(); qtbot.addWidget(w)
    w.load_profile(_profile())
    out = tmp_path / "run.sh"
    w.export_sh(str(out))
    text = out.read_text()
    assert text.startswith("#!/usr/bin/env bash")
    assert "podman run --rm" in text
    assert os.access(out, os.X_OK)


def test_model_meta_text(qtbot, monkeypatch):
    monkeypatch.setattr(mw.model_info, "read_gguf_meta",
                        lambda path: GgufMeta(quant="IQ3_S", size_label="30B-A3B"))
    monkeypatch.setattr(mw.model_info, "file_size", lambda path: 15 * 1024**3)
    w = mw.MainWindow(); qtbot.addWidget(w)
    w.load_profile(_profile())
    t = w.model_meta_text()
    assert "IQ3_S" in t and "30B-A3B" in t and "GiB" in t


def test_lora_browse_resolver(qtbot):
    panel = LoraPanel()
    qtbot.addWidget(panel)
    panel.set_browse_resolver(lambda host: "/models/" + host.split("/")[-1])
    panel.set_loras([LoraRef(path="", scale=1.0)])
    # simulate the row's browse handler resolving a picked host path
    assert panel._resolve("/mnt/Models/a.gguf") == "/models/a.gguf"
