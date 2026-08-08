import pytest

from llama_launcher.ui.main_window import MainWindow


@pytest.fixture
def win(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def test_fetch_button_has_tooltip(win):
    assert win.fetch_btn.toolTip().strip()
    # the tooltip must note the not-downloaded caveat
    assert "pull" in win.fetch_btn.toolTip().lower()


def test_fetch_button_sits_next_to_detect(win):
    # Both act on the Image field, so they share the Image row's parent widget.
    assert win.fetch_btn.parentWidget() is win.detect_image_btn.parentWidget()


def test_fetch_button_relabeled(win):
    assert win.fetch_btn.text() == "Fetch latest"
