from llama_launcher.core.settings_catalog import CATALOG
from llama_launcher.ui.widgets.setting_widgets import make_widget

LADDER = (
    0,
    1024,
    2048,
    4096,
    8192,
    12288,
    16384,
    24576,
    32768,
    49152,
    65536,
    98304,
    131072,
    196608,
    262144,
)


def test_ctx_size_offers_presets_and_accepts_custom(qtbot):
    w = make_widget(CATALOG["ctx-size"])
    qtbot.addWidget(w)
    w.set_value(32768)  # pick a preset
    assert w.value() == 32768
    w.set_value(0)  # 0 = model default => unset
    assert w.is_set() is False


def test_ctx_size_ladder_is_exact():
    assert CATALOG["ctx-size"].suggestions == LADDER


def test_ctx_size_combo_lists_the_ladder_in_order_and_stays_editable(qtbot):
    w = make_widget(CATALOG["ctx-size"])
    qtbot.addWidget(w)
    items = [w._editor.itemText(i) for i in range(w._editor.count())]
    assert items == [str(v) for v in LADDER]
    assert w._editor.isEditable()
    w._editor.setCurrentText("50000")
    assert w.value() == 50000
