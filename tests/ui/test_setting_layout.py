from PySide6.QtWidgets import QGridLayout

from llama_launcher.core.settings_catalog import CATALOG
from llama_launcher.ui.main_window import MainWindow
from llama_launcher.ui.widgets.setting_widgets import make_widget


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


def test_tools_checkboxes_in_grid(qtbot):
    w = make_widget(CATALOG["tools"])
    qtbot.addWidget(w)
    layout = w._editor.layout()
    assert isinstance(layout, QGridLayout)
    assert layout.columnCount() == 3          # 3x3 grid
    assert layout.count() == 9                # "all" + 8 built-in tools
