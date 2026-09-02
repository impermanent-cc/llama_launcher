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
    # A changed slot that reads loras() must not crash on a half-built row.
    panel = LoraPanel()
    qtbot.addWidget(panel)
    panel.changed.connect(panel.loras)
    panel._add_row(LoraRef(path="/loras/a.gguf", scale=0.5))
    out = panel.loras()
    assert len(out) == 1
    assert out[0].path == "/loras/a.gguf"


def test_scale_change_emits_zero_arg_changed(qtbot):
    # valueChanged passes an arg; the 0-arg changed signal must still fire.
    panel = LoraPanel()
    qtbot.addWidget(panel)
    panel.set_loras([LoraRef(path="/loras/a.gguf", scale=1.0)])
    count = {"n": 0}
    panel.changed.connect(lambda: count.__setitem__("n", count["n"] + 1))
    panel.table.cellWidget(0, 1).setValue(0.5)
    assert count["n"] >= 1
    assert abs(panel.loras()[0].scale - 0.5) < 1e-6


# -- live scale control ------------------------------------------------------

from llama_launcher.core.lora_state import LoraAdapter
from llama_launcher.services import lora_api


def _panel_with_target(qtbot, target=("127.0.0.1", 8080, None)):
    panel = LoraPanel()
    qtbot.addWidget(panel)
    panel.set_live_resolver(lambda: target)
    return panel


def test_live_buttons_disabled_without_a_target(qtbot):
    panel = LoraPanel()
    qtbot.addWidget(panel)
    assert not panel.apply_btn.isEnabled()
    panel.set_live_resolver(lambda: None)
    assert not panel.apply_btn.isEnabled()
    panel.set_live_resolver(lambda: ("h", 1, None))
    assert panel.apply_btn.isEnabled()


def test_a_raising_resolver_does_not_break_the_form(qtbot):
    panel = LoraPanel()
    qtbot.addWidget(panel)

    def boom():
        raise RuntimeError("no profile yet")

    panel.set_live_resolver(boom)
    assert not panel.apply_btn.isEnabled()


def test_sync_loads_server_scales_into_matching_rows(qtbot, monkeypatch):
    panel = _panel_with_target(qtbot)
    panel.set_loras([LoraRef(path="/l/a.gguf", scale=1.0),
                     LoraRef(path="/l/b.gguf", scale=1.0)])
    monkeypatch.setattr(lora_api, "list_adapters", lambda *a, **kw: [
        LoraAdapter(id=0, path="/l/a.gguf", scale=0.25),
        LoraAdapter(id=1, path="/l/b.gguf", scale=0.0),
    ])
    panel.sync_from_server()
    qtbot.waitUntil(lambda: "matched" in panel.live_status.text(), timeout=3000)
    assert abs(panel.loras()[0].scale - 0.25) < 1e-6
    # A zeroed adapter is loaded-but-inactive, and the row must show 0, not 1.
    assert abs(panel.table.cellWidget(1, 1).value()) < 1e-6
    assert "1 active" in panel.live_status.text()


def test_sync_reports_unreachable_without_touching_rows(qtbot, monkeypatch):
    panel = _panel_with_target(qtbot)
    panel.set_loras([LoraRef(path="/l/a.gguf", scale=0.4)])
    monkeypatch.setattr(lora_api, "list_adapters", lambda *a, **kw: None)
    panel.sync_from_server()
    qtbot.waitUntil(lambda: "Could not reach" in panel.live_status.text(), timeout=3000)
    assert abs(panel.loras()[0].scale - 0.4) < 1e-6


def test_sync_explains_a_server_launched_without_adapters(qtbot, monkeypatch):
    panel = _panel_with_target(qtbot)
    monkeypatch.setattr(lora_api, "list_adapters", lambda *a, **kw: [])
    panel.sync_from_server()
    qtbot.waitUntil(lambda: "no LoRA adapters" in panel.live_status.text(), timeout=3000)


def test_apply_sends_every_loaded_adapter_keyed_by_live_id(qtbot, monkeypatch):
    panel = _panel_with_target(qtbot)
    panel.set_loras([LoraRef(path="/l/b.gguf", scale=0.75)])
    monkeypatch.setattr(lora_api, "list_adapters", lambda *a, **kw: [
        LoraAdapter(id=4, path="/l/a.gguf", scale=0.9),
        LoraAdapter(id=7, path="/l/b.gguf", scale=0.0),
    ])
    seen = {}

    def fake_set(host, port, key, scales, timeout=None):
        seen["scales"] = scales
        return True

    monkeypatch.setattr(lora_api, "set_scales", fake_set)
    panel.apply_to_server()
    qtbot.waitUntil(lambda: "Applied" in panel.live_status.text(), timeout=3000)
    # Row matched by PATH resolves to the server's own id (7, not row index 0),
    # and the unlisted adapter keeps the scale the server already reports
    # instead of being silently zeroed.
    assert seen["scales"] == {4: 0.9, 7: 0.75}
    assert "1 adapter(s) rescaled" in panel.live_status.text()


def test_apply_reports_a_rejected_change(qtbot, monkeypatch):
    panel = _panel_with_target(qtbot)
    panel.set_loras([LoraRef(path="/l/a.gguf", scale=0.5)])
    monkeypatch.setattr(lora_api, "list_adapters", lambda *a, **kw: [
        LoraAdapter(id=0, path="/l/a.gguf", scale=0.0)])
    monkeypatch.setattr(lora_api, "set_scales", lambda *a, **kw: False)
    panel.apply_to_server()
    qtbot.waitUntil(lambda: "rejected" in panel.live_status.text(), timeout=3000)


def test_apply_with_no_rows_does_not_call_the_server(qtbot, monkeypatch):
    panel = _panel_with_target(qtbot)
    called = {"n": 0}
    monkeypatch.setattr(lora_api, "list_adapters",
                        lambda *a, **kw: called.__setitem__("n", called["n"] + 1))
    panel.apply_to_server()
    assert called["n"] == 0
    assert "no adapter rows" in panel.live_status.text()
