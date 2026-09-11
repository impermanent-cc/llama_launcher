import re

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from llama_launcher.core.spec import Profile
from llama_launcher.ui.widgets import setting_search as ss


def _entry(*texts, visible=True, kind="setting"):
    return ss.SearchEntry(texts, kind, object(), lambda: visible)


def test_match_positions_is_a_dash_insensitive_substring_in_order():
    entries = [
        _entry("--ctx-size", "-c"),
        _entry("Sampling", kind="group"),
        _entry("--n-gpu-layers", "-ngl"),
        _entry("--ctx-checkpoints"),
    ]
    assert ss.match_positions("ctx", entries) == [0, 3]
    assert ss.match_positions("--CTX", entries) == [0, 3]
    assert ss.match_positions("ngl", entries) == [2]
    assert ss.match_positions("samp", entries) == [1]
    assert ss.match_positions("", entries) == []
    assert ss.match_positions("   ", entries) == []


def test_hidden_entries_never_match():
    entries = [_entry("--ctx-size", visible=False), _entry("--ctx-checkpoints")]
    assert ss.match_positions("ctx", entries) == [1]


def test_typing_jumps_to_the_first_match_and_tints_its_label(main_window):
    panel = main_window._configure_panel
    bar = panel.search_bar
    QTest.keyClicks(bar.edit, "ctx-size")
    assert re.fullmatch(r"1 of \d+", bar.counter.text())
    label = panel._setting_rows["ctx-size"][0].labelForField(panel._widgets["ctx-size"])
    assert label.styleSheet()


def test_enter_cycles_and_shift_enter_goes_back(main_window):
    """Enter steps forward through the matches and wraps from the last back
    to the first; Shift+Enter steps backward and wraps from the first to
    the last."""
    panel = main_window._configure_panel
    bar = panel.search_bar
    QTest.keyClicks(bar.edit, "ctx")
    first = bar.current()
    total = int(re.fullmatch(r"1 of (\d+)", bar.counter.text()).group(1))
    QTest.keyClick(bar.edit, Qt.Key_Return)
    assert bar.current() is not first and bar.counter.text().startswith("2 of")
    assert not first.widget.styleSheet()
    for _ in range(total - 2):
        QTest.keyClick(bar.edit, Qt.Key_Return)
    assert bar.counter.text() == f"{total} of {total}"
    QTest.keyClick(bar.edit, Qt.Key_Return)
    assert bar.counter.text() == f"1 of {total}"
    assert bar.current() is first
    QTest.keyClick(bar.edit, Qt.Key_Return, Qt.ShiftModifier)
    assert bar.counter.text() == f"{total} of {total}"


def test_escape_clears_the_field_the_counter_and_the_tint(main_window):
    panel = main_window._configure_panel
    bar = panel.search_bar
    QTest.keyClicks(bar.edit, "ctx")
    target = bar.current().widget
    QTest.keyClick(bar.edit, Qt.Key_Escape)
    assert (
        bar.edit.text() == "" and bar.counter.text() == "" and not target.styleSheet()
    )


def test_a_group_match_tints_the_group_title(main_window):
    panel = main_window._configure_panel
    bar = panel.search_bar
    QTest.keyClicks(bar.edit, "sampl")
    assert bar.current().kind == "group"
    assert "QGroupBox::title" in panel._group_boxes["Sampling"].styleSheet()


def test_rows_hidden_by_the_mode_do_not_match(main_window):
    panel = main_window._configure_panel
    panel.load_profile(Profile(name="Host", mode="router", image="img"))
    bar = panel.search_bar
    QTest.keyClicks(bar.edit, "ctx-size")
    assert bar.counter.text() == "no match"


def test_a_group_hidden_entirely_by_the_mode_gives_no_match(main_window):
    """A router profile hides every row of the Sampling group, so its title
    no longer matches either."""
    panel = main_window._configure_panel
    panel.load_profile(Profile(name="Host", mode="router", image="img"))
    bar = panel.search_bar
    QTest.keyClicks(bar.edit, "sampl")
    assert bar.counter.text() == "no match"


def test_a_mode_switch_refreshes_an_already_typed_query(main_window):
    """Loading a profile that hides ctx-size's row turns a query typed
    beforehand into no match, without the query being retyped."""
    panel = main_window._configure_panel
    bar = panel.search_bar
    QTest.keyClicks(bar.edit, "ctx-size")
    panel.load_profile(Profile(name="Host", mode="router", image="img"))
    assert bar.counter.text() == "no match"


def test_jumping_to_a_late_group_scrolls_the_settings_box(main_window, qtbot):
    panel = main_window._configure_panel
    with qtbot.waitExposed(main_window):
        main_window.show()
    bar = panel.search_bar
    before = panel._right_scroll.verticalScrollBar().value()
    QTest.keyClicks(bar.edit, "speculative")
    assert bar.current().kind == "group"
    assert panel._right_scroll.verticalScrollBar().value() != before


def test_refresh_keeps_the_current_entry_selected_across_an_engine_switch(main_window):
    """Switching engine recomputes the matches but does not reset the
    selection to the first one when the previously selected entry still
    matches."""
    panel = main_window._configure_panel
    bar = panel.search_bar
    QTest.keyClicks(bar.edit, "gpu-layers")
    bar.next_match()
    kept = bar.current()
    other_index = 1 - panel.engine_combo.currentIndex()
    panel.engine_combo.setCurrentIndex(other_index)
    assert bar.current() is kept


def test_a_bare_dash_query_reads_as_blank(main_window):
    """A query that is only dashes normalizes to nothing, so it behaves like
    an empty field: no counter text and nothing tinted."""
    panel = main_window._configure_panel
    bar = panel.search_bar
    QTest.keyClicks(bar.edit, "--")
    assert bar.counter.text() == ""
    assert panel._tinted is None


def test_a_jump_never_moves_focus_into_a_setting(main_window, qtbot):
    with qtbot.waitExposed(main_window):
        main_window.show()
    panel = main_window._configure_panel
    bar = panel.search_bar
    QTest.keyClicks(bar.edit, "ctx-size")
    QTest.keyClick(bar.edit, Qt.Key_Return)
    qtbot.wait(10)
    assert not panel._widgets["ctx-size"].hasFocus()


def test_the_bar_sits_above_the_settings_scroll_area(main_window):
    panel = main_window._configure_panel
    col = panel.search_bar.parentWidget().layout()
    assert col.indexOf(panel.search_bar) < col.indexOf(panel._right_scroll)


def test_a_setting_tooltip_is_not_searched(main_window):
    panel = main_window._configure_panel
    bar = panel.search_bar
    QTest.keyClicks(bar.edit, "trained")
    assert bar.counter.text() == "no match"
