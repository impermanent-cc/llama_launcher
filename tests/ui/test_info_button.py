from llama_launcher.ui.widgets.info_button import InfoButton


def test_carries_text_and_hover(qtbot):
    b = InfoButton("gen/prompt = tok/s of the last request")
    qtbot.addWidget(b)
    assert b.info_text == "gen/prompt = tok/s of the last request"
    assert b.toolTip() == b.info_text          # hover still works
    assert "\U0001f4a1" not in b.text() and b.text() == "\u24d8"   # non-emoji glyph


def test_click_does_not_crash(qtbot):
    b = InfoButton("hello")
    qtbot.addWidget(b)
    b.click()                                   # opens popover; must not raise
