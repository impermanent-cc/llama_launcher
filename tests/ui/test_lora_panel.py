from llama_launcher.core.spec import LoraRef
from llama_launcher.ui.panels.lora_panel import LoraPanel


def test_lora_panel_roundtrip(qtbot):
    panel = LoraPanel()
    qtbot.addWidget(panel)
    loras = [LoraRef(path="/loras/a.gguf", scale=1.0),
             LoraRef(path="/loras/b.gguf", scale=0.5)]
    panel.set_loras(loras)
    out = panel.loras()
    assert len(out) == 2
    assert out[0].path == "/loras/a.gguf" and abs(out[0].scale - 1.0) < 1e-6
    assert out[1].path == "/loras/b.gguf" and abs(out[1].scale - 0.5) < 1e-6


def test_lora_panel_skips_empty_path_rows(qtbot):
    panel = LoraPanel()
    qtbot.addWidget(panel)
    panel.set_loras([LoraRef(path="", scale=0.7),
                     LoraRef(path="/loras/c.gguf", scale=0.3)])
    out = panel.loras()
    assert len(out) == 1
    assert out[0].path == "/loras/c.gguf" and abs(out[0].scale - 0.3) < 1e-6
