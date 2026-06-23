from llama_launcher.core.settings_catalog import CATALOG
from llama_launcher.ui.widgets.setting_widgets import make_widget


def test_ctx_size_offers_presets_and_accepts_custom(qtbot):
    w = make_widget(CATALOG["ctx-size"])
    qtbot.addWidget(w)
    items = [w._editor.itemText(i) for i in range(w._editor.count())]
    assert "65536" in items and "131072" in items     # presets present
    w.set_value(32768)                                 # pick a preset
    assert w.value() == 32768
    w._editor.setCurrentText("50000")                  # type a custom value
    assert w.value() == 50000
    w.set_value(0)                                     # 0 = model default => unset
    assert w.is_set() is False
