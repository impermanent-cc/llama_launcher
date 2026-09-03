from llama_launcher.core.settings_catalog import CATALOG
from llama_launcher.ui.widgets.setting_widgets import make_row_label, make_widget


def test_row_label_plain_for_normal_setting(qtbot):
    lbl = make_row_label(CATALOG["ctx-size"])
    qtbot.addWidget(lbl)
    assert lbl.text() == "--ctx-size"


def test_row_label_marks_deprecated_setting(qtbot):
    lbl = make_row_label(CATALOG["no-mmap"])
    qtbot.addWidget(lbl)
    assert "--no-mmap" in lbl.text()
    assert "*deprecated" in lbl.text()
    # the replacement is discoverable on hover
    assert "load-mode" in lbl.toolTip()


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
    assert tip.startswith("\u26a0 DANGER:")


def test_danger_border_scoped(qtbot):
    w = make_widget(CATALOG["tools"])
    qtbot.addWidget(w)
    ss = w.styleSheet()
    assert "#dangerSetting" in ss
    assert "red" in ss


def test_non_danger_widget_no_stylesheet(qtbot):
    w = make_widget(CATALOG["temp"])
    qtbot.addWidget(w)
    assert w.styleSheet() == ""


def test_danger_border_does_not_cascade_to_checkboxes(qtbot):
    w = make_widget(CATALOG["tools"])
    qtbot.addWidget(w)
    for cb in w._checks.values():
        assert cb.styleSheet() == ""


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


def test_numa_off_sentinel_is_not_set(qtbot):
    from llama_launcher.core.settings_catalog import CATALOG
    from llama_launcher.ui.widgets.setting_widgets import make_widget

    w = make_widget(CATALOG["numa"])
    qtbot.addWidget(w)
    w.set_value("off")
    assert w.is_set() is False  # equals default -> not stored/emitted
    w.set_value("distribute")
    assert w.is_set() is True


def test_set_enum_choices_extends_and_reverts(qtbot):
    w = make_widget(CATALOG["cache-type-k"])  # enum, default "f16"
    qtbot.addWidget(w)
    base = list(CATALOG["cache-type-k"].enum)

    w.set_enum_choices([*base, "q6_0", "q8_KV"])
    w.set_value("q6_0")
    assert w.value() == "q6_0"

    # Reverting to base drops the now-invalid selection back to the default.
    w.set_enum_choices(base)
    assert w.value() == "f16"


def test_set_enum_choices_noop_on_non_enum(qtbot):
    w = make_widget(CATALOG["ctx-size"])  # int
    qtbot.addWidget(w)
    w.set_enum_choices(["a", "b"])  # must not raise
    assert w.value() == CATALOG["ctx-size"].default


def test_api_key_widget_is_password_masked(qtbot):
    from PySide6.QtWidgets import QLineEdit

    w = make_widget(CATALOG["api-key"])
    w.set_value("sk-secret")
    assert w._editor.echoMode() == QLineEdit.Password
    assert w.value() == "sk-secret"  # value still readable programmatically


def test_bool_widget_initializes_to_setting_default(qtbot):
    # The build catalog has default-True bools; an unchecked box for a
    # default-ON option would read as "explicitly OFF".
    from llama_launcher.core.build_catalog import BUILD_CATALOG

    w = make_widget(BUILD_CATALOG["cuda-fa"])  # GGML_CUDA_FA, default True
    qtbot.addWidget(w)
    assert w.value() is True
    assert w.is_set() is False


def test_string_widget_initializes_to_setting_default(qtbot):
    # Non-empty-default strings exist in BOTH catalogs (build: blas-vendor
    # "Generic"; runtime: cors-origins "*"). A fresh empty editor must not
    # read as "explicitly set to blank": is_set() True with value "" pollutes
    # saved configs/profiles with phantom entries and makes the router's
    # wildcard-CORS warning unfireable from UI-saved profiles.
    from llama_launcher.core.build_catalog import BUILD_CATALOG
    from llama_launcher.core.settings_catalog import CATALOG

    for setting in (BUILD_CATALOG["blas-vendor"], CATALOG["cors-origins"]):
        w = make_widget(setting)
        qtbot.addWidget(w)
        assert w.value() == setting.default
        assert w.is_set() is False
