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


def test_lora_panel_item_changed_emits_signal(qtbot):
    panel = LoraPanel()
    qtbot.addWidget(panel)
    panel.set_loras([LoraRef(path="/loras/a.gguf", scale=1.0)])
    with qtbot.waitSignal(panel.changed, timeout=1000):
        panel.table.item(0, 0).setText("/new/path.gguf")


def test_add_row_does_not_crash_when_changed_reads_loras(qtbot):
    # Regression for Bug 3.
    panel = LoraPanel()
    qtbot.addWidget(panel)
    panel.changed.connect(panel.loras)
    panel._add_row(LoraRef(path="/loras/a.gguf", scale=0.5))
    out = panel.loras()
    assert len(out) == 1
    assert out[0].path == "/loras/a.gguf"


def test_scale_change_emits_zero_arg_changed(qtbot):
    # Regression for Bug 4: valueChanged passes an arg to the 0-arg signal.
    panel = LoraPanel()
    qtbot.addWidget(panel)
    panel.set_loras([LoraRef(path="/loras/a.gguf", scale=1.0)])
    count = {"n": 0}
    panel.changed.connect(lambda: count.__setitem__("n", count["n"] + 1))
    panel.table.cellWidget(0, 1).setValue(0.5)
    assert count["n"] >= 1
    assert abs(panel.loras()[0].scale - 0.5) < 1e-6
