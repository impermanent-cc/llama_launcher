import pytest
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QSystemTrayIcon
import llama_launcher.ui.main_window as mw


@pytest.fixture(autouse=True)
def _tray_available(monkeypatch):
    # Most tests want a tray available; individual tests may override.
    monkeypatch.setattr(
        "PySide6.QtWidgets.QSystemTrayIcon.isSystemTrayAvailable",
        staticmethod(lambda: True),
    )


def test_close_hides_to_tray(qtbot):
    w = mw.MainWindow(); qtbot.addWidget(w); w.show()
    assert w.tray is not None
    ev = QCloseEvent()
    w.closeEvent(ev)
    assert not ev.isAccepted()      # close was intercepted (hidden to tray)
    assert not w.isVisible()


def test_quit_app_accepts_close(qtbot):
    w = mw.MainWindow(); qtbot.addWidget(w)
    w._really_quit = True
    ev = QCloseEvent()
    w.closeEvent(ev)
    assert ev.isAccepted()


def test_no_tray_sets_tray_none(qtbot, monkeypatch):
    monkeypatch.setattr(
        "PySide6.QtWidgets.QSystemTrayIcon.isSystemTrayAvailable",
        staticmethod(lambda: False),
    )
    w = mw.MainWindow(); qtbot.addWidget(w)
    assert w.tray is None
    assert w._tray_enabled is False


def test_no_tray_close_event_accepts(qtbot, monkeypatch):
    monkeypatch.setattr(
        "PySide6.QtWidgets.QSystemTrayIcon.isSystemTrayAvailable",
        staticmethod(lambda: False),
    )
    w = mw.MainWindow(); qtbot.addWidget(w)
    ev = QCloseEvent()
    w.closeEvent(ev)
    assert ev.isAccepted()
