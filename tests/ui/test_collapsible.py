from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from llama_launcher.ui.widgets.collapsible import CollapsibleSection


def test_collapsed_by_default(qtbot):
    content = QLabel("hi")
    sec = CollapsibleSection("Title", content)
    qtbot.addWidget(sec)
    assert sec.is_expanded() is False
    assert content.isVisible() is False


def test_set_expanded_shows_content(qtbot):
    content = QLabel("hi")
    sec = CollapsibleSection("Title", content)
    qtbot.addWidget(sec)
    sec.show()
    sec.set_expanded(True)
    assert sec.is_expanded() is True
    assert content.isVisible() is True


def test_toggle_hides_again(qtbot):
    content = QLabel("hi")
    sec = CollapsibleSection("Title", content)
    qtbot.addWidget(sec)
    sec.show()
    sec.set_expanded(True)
    assert content.isVisible() is True
    sec.set_expanded(False)
    assert sec.is_expanded() is False
    assert content.isVisible() is False


def test_arrow_reflects_state(qtbot):
    content = QLabel("hi")
    sec = CollapsibleSection("Title", content)
    qtbot.addWidget(sec)
    assert sec._button.arrowType() == Qt.RightArrow
    sec.set_expanded(True)
    assert sec._button.arrowType() == Qt.DownArrow


def test_start_expanded(qtbot):
    content = QLabel("hi")
    sec = CollapsibleSection("Title", content, collapsed=False)
    qtbot.addWidget(sec)
    sec.show()
    assert sec.is_expanded() is True
    assert content.isVisible() is True
