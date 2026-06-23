from PySide6.QtGui import QCloseEvent
import llama_launcher.ui.main_window as mw


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
