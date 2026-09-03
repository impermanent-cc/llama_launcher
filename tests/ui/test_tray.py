import pytest
from PySide6.QtGui import QCloseEvent

import llama_launcher.ui.main_window as mw


@pytest.fixture(autouse=True)
def _tray_available(monkeypatch):
    # Default: a tray IS available, to prove that tray availability ALONE does
    # not enable minimize-to-tray (the user must also opt in via config).
    monkeypatch.setattr(
        "PySide6.QtWidgets.QSystemTrayIcon.isSystemTrayAvailable",
        staticmethod(lambda: True),
    )


def _opt_in(monkeypatch):
    monkeypatch.setattr(mw, "load_config", lambda *a, **k: {"minimize_to_tray": True})


def test_close_quits_by_default_even_with_tray(qtbot):
    # tray available but minimize-to-tray NOT opted in -> close quits.
    w = mw.MainWindow()
    qtbot.addWidget(w)
    assert w._minimize_to_tray is False
    assert w.tray is None
    ev = QCloseEvent()
    w.closeEvent(ev)
    assert ev.isAccepted()


def test_close_hides_to_tray_when_opted_in(qtbot, monkeypatch):
    _opt_in(monkeypatch)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w.show()
    assert w._minimize_to_tray is True
    assert w.tray is not None
    ev = QCloseEvent()
    w.closeEvent(ev)
    assert not ev.isAccepted()  # close intercepted (hidden to tray)
    assert not w.isVisible()


def test_quit_app_accepts_close_when_opted_in(qtbot, monkeypatch):
    _opt_in(monkeypatch)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._really_quit = True
    ev = QCloseEvent()
    w.closeEvent(ev)
    assert ev.isAccepted()


def test_opt_in_but_no_tray_disables_minimize(qtbot, monkeypatch):
    _opt_in(monkeypatch)  # opted in...
    monkeypatch.setattr(
        "PySide6.QtWidgets.QSystemTrayIcon.isSystemTrayAvailable",
        staticmethod(lambda: False),
    )  # ...but no usable tray -> minimize-to-tray stays off, close quits
    w = mw.MainWindow()
    qtbot.addWidget(w)
    assert w.tray is None
    assert w._minimize_to_tray is False
    ev = QCloseEvent()
    w.closeEvent(ev)
    assert ev.isAccepted()
