from llama_launcher.core.settings_catalog import CATALOG
from llama_launcher.ui.widgets.setting_widgets import make_widget


def test_bool_widget_roundtrip(qtbot):
    w = make_widget(CATALOG["no-mmap"])
    qtbot.addWidget(w)
    w.set_value(True)
    assert w.value() is True
    assert w.is_set() is True


def test_enum_widget_roundtrip(qtbot):
    w = make_widget(CATALOG["flash-attn"])
    qtbot.addWidget(w)
    w.set_value("on")
    assert w.value() == "on"


def test_int_widget_roundtrip(qtbot):
    w = make_widget(CATALOG["ctx-size"])
    qtbot.addWidget(w)
    w.set_value(65536)
    assert w.value() == 65536


def test_float_widget_roundtrip(qtbot):
    w = make_widget(CATALOG["temp"])
    qtbot.addWidget(w)
    w.set_value(0.6)
    assert abs(w.value() - 0.6) < 1e-6


def test_int_or_token_widget(qtbot):
    w = make_widget(CATALOG["n-gpu-layers"])
    qtbot.addWidget(w)
    w.set_value("all")
    assert w.value() == "all"
    w.set_value(99)
    assert w.value() == 99


def test_int_or_token_empty_returns_default(qtbot):
    w = make_widget(CATALOG["n-gpu-layers"])
    qtbot.addWidget(w)
    w.set_value("")
    assert w.value() == CATALOG["n-gpu-layers"].default
    assert w.is_set() is False


def test_danger_widget_tooltip(qtbot):
    w = make_widget(CATALOG["tools"])
    qtbot.addWidget(w)
    tip = w.toolTip()
    assert tip
    assert tip.startswith("⚠ DANGER:")


def test_normal_widget_tooltip(qtbot):
    w = make_widget(CATALOG["temp"])
    qtbot.addWidget(w)
    assert w.toolTip() == CATALOG["temp"].tooltip


def test_tools_multiselect_all(qtbot):
    w = make_widget(CATALOG["tools"])
    qtbot.addWidget(w)
    w.set_value("all")
    assert w.value() == "all"


def test_tools_multiselect_subset(qtbot):
    w = make_widget(CATALOG["tools"])
    qtbot.addWidget(w)
    w.set_value("read_file,grep_search")
    assert set(w.value().split(",")) == {"read_file", "grep_search"}


def test_tools_multiselect_default_not_set(qtbot):
    w = make_widget(CATALOG["tools"])
    qtbot.addWidget(w)
    assert w.value() == ""
    assert w.is_set() is False


def test_tools_multiselect_all_disables_individual(qtbot):
    w = make_widget(CATALOG["tools"])
    qtbot.addWidget(w)
    w.set_value("all")
    assert all(not cb.isEnabled() for cb in w._checks.values())
    w.set_value("")
    assert all(cb.isEnabled() for cb in w._checks.values())
