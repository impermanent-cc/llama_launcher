import pytest

from llama_launcher.ui.main_window import MainWindow


@pytest.fixture
def win(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def test_window_minimum_height_fits_1080p(win, qtbot):
    win.show()
    win.layout().activate()
    # Before the fix the minimum height is ~959 (taller than a 1080p screen with a
    # taskbar can comfortably show). After wrapping the left column it must drop well
    # below that so the window is resizable to fit.
    assert win.minimumSizeHint().height() <= 720


def test_window_can_shrink_below_the_old_floor(win, qtbot):
    win.show()
    win.resize(1000, 700)
    qtbot.wait(10)
    assert win.height() <= 760, "window should hold a short height, not spring back to ~959"
