from PySide6.QtWidgets import QApplication

from llama_launcher.ui import main_window as mw
from llama_launcher.ui.main_window import MainWindow
from llama_launcher.ui.panels import configure_panel as cp


def _shown(qtbot, width=1600, height=900):
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(width, height)
    w.show()
    QApplication.processEvents()
    return w


def _resize(w, width, height=900):
    w.resize(width, height)
    QApplication.processEvents()


def test_environment_column_grows_to_its_maximum_then_stops(qtbot):
    w = _shown(qtbot)
    left = w._configure_panel._left_scroll
    right = w._configure_panel._right_scroll
    assert (cp.ENV_COLUMN_MIN, cp.ENV_COLUMN_MAX) == (420, 640)
    assert left.width() == 640
    right_at_1600 = right.width()
    _resize(w, 2560)
    assert left.width() == 640
    assert right.width() >= right_at_1600 + 900


def test_environment_column_shares_a_narrow_window_inside_its_bounds(qtbot):
    w = _shown(qtbot, width=1100)
    left = w._configure_panel._left_scroll
    assert 420 <= left.width() < 640


def test_top_bar_fields_grow_only_to_their_maximum(qtbot):
    w = _shown(qtbot)
    name = w._configure_panel.name_edit
    combo = w._configure_panel.profile_combo
    assert mw.NAME_EDIT_BOUNDS == (160, 260)
    assert mw.PROFILE_COMBO_BOUNDS == (200, 340)
    assert 160 < name.width() <= 260
    assert 200 < combo.width() <= 340
    _resize(w, 2560)
    assert name.width() == 260
    assert combo.width() == 340
    _resize(w, 1100)
    assert 160 <= name.width() <= 260
    assert 200 <= combo.width() <= 340


def test_top_bar_status_label_sits_after_a_stretch(qtbot):
    w = _shown(qtbot)
    bar = w._top_bar
    items = [bar.itemAt(i) for i in range(bar.count())]
    assert items[-1].widget() is w.status_label
    assert items[-2].spacerItem() is not None
    assert items[-3].widget() is w.stats_toggle_btn
    _resize(w, 2560)
    gap = w.status_label.x() - (w.stats_toggle_btn.x() + w.stats_toggle_btn.width())
    assert gap > 900
