from PySide6.QtWidgets import QGridLayout

from llama_launcher.core.settings_catalog import CATALOG
from llama_launcher.ui.main_window import MainWindow
from llama_launcher.ui.widgets.setting_widgets import (
    DOT_WIDTH,
    MULTISELECT_COLUMNS,
    make_widget,
)


def _row_label(panel, key):
    form, widget = panel._setting_rows[key]
    return form.labelForField(widget)


def test_deprecated_row_shows_marker_in_panel(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    panel = w._configure_panel
    assert "*deprecated" in _row_label(panel, "no-mmap").text()
    assert "*deprecated" not in _row_label(panel, "ctx-size").text()


def test_editor_widths_are_capped(qtbot):
    # enum/int/float/string/int_or_token editors must not stretch full-width
    for key in ["flash-attn", "ctx-size", "temp", "api-key", "n-gpu-layers"]:
        w = make_widget(CATALOG[key])
        qtbot.addWidget(w)
        assert 0 < w._editor.maximumWidth() < 1000, key


def test_tools_checkboxes_in_two_columns(qtbot):
    w = make_widget(CATALOG["tools"])
    qtbot.addWidget(w)
    layout = w._editor.layout()
    assert isinstance(layout, QGridLayout)
    assert MULTISELECT_COLUMNS == 2
    assert layout.columnCount() == 2
    assert layout.rowCount() == 4
    assert layout.count() == 8  # "all" + 7 built-in tools
    assert layout.itemAtPosition(0, 0).widget() is w._all_check
    assert layout.itemAtPosition(0, 1).widget() is w._checks["read_file"]
    assert layout.itemAtPosition(3, 1).widget() is w._checks["get_info"]


def test_dot_sits_directly_after_the_editor_before_the_stretch(qtbot):
    for key in ["ctx-size", "flash-attn", "temp", "api-key", "tools"]:
        w = make_widget(CATALOG[key])
        qtbot.addWidget(w)
        layout = w.layout()
        items = [layout.itemAt(i) for i in range(layout.count())]
        dot_index = next(i for i, it in enumerate(items) if it.widget() is w._dot)
        before = items[dot_index - 1].widget()
        assert before is not None, key
        assert before is w._editor or before is getattr(w, "_reveal_btn", None), key
        assert items[dot_index + 1].spacerItem() is not None, key
        assert dot_index + 2 == len(items), key


def test_dot_is_sixteen_pixels_wide(qtbot):
    w = make_widget(CATALOG["ctx-size"])
    qtbot.addWidget(w)
    assert DOT_WIDTH == 16
    assert w._dot.minimumWidth() == 16 and w._dot.maximumWidth() == 16


def test_settings_column_minimum_width_stays_under_600(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    inner = w._configure_panel._right_scroll.widget()
    assert inner.minimumSizeHint().width() < 600
