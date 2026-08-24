from llama_launcher.core.settings_catalog import CATALOG
from llama_launcher.ui.widgets.setting_widgets import SuggestionDot, make_widget


def test_dot_states(qtbot):
    d = SuggestionDot()
    qtbot.addWidget(d); d.show()
    d.set_state("none")
    assert not d.isVisible() or d.text() == ""
    d.set_state("suggested", "MoE model: offload experts to CPU")
    assert d.isVisible() and d.text() == "●"
    assert d.toolTip() == "MoE model: offload experts to CPU"
    d.set_state("muted", "Not a MoE model")
    assert d.text() == "○"


def test_dot_click_applies(qtbot):
    d = SuggestionDot()
    qtbot.addWidget(d)
    calls = []
    d.set_state("suggested", "set flash-attn", on_apply=lambda: calls.append(1))
    d.click()
    assert calls == [1]


def test_dot_without_on_apply_is_passive(qtbot):
    d = SuggestionDot()
    qtbot.addWidget(d)
    d.set_state("suggested", "recommended", on_apply=None)
    d.click()                      # no callback wired; must not raise
    assert not d.isEnabled()       # passive indicator


def test_setting_widget_exposes_set_suggestion(qtbot):
    w = make_widget(CATALOG["flash-attn"])
    qtbot.addWidget(w)
    w.set_suggestion("suggested", "Enable flash attention")
    assert w._dot.text() == "●"
