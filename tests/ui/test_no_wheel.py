from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent

from llama_launcher.ui.widgets.no_wheel import (
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
)


def _wheel_event():
    pos = QPointF(5.0, 5.0)
    return QWheelEvent(
        pos,
        pos,
        QPoint(0, 0),
        QPoint(0, -120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.ScrollUpdate,
        False,
    )


def test_combobox_ignores_wheel_when_unfocused(qtbot):
    w = NoWheelComboBox()
    w.addItems(["a", "b", "c"])
    w.setCurrentIndex(1)
    qtbot.addWidget(w)
    assert not w.hasFocus()
    before = w.currentIndex()
    event = _wheel_event()
    w.wheelEvent(event)
    assert w.currentIndex() == before
    assert event.isAccepted() is False


def test_spinbox_ignores_wheel_when_unfocused(qtbot):
    w = NoWheelSpinBox()
    w.setRange(0, 100)
    w.setValue(10)
    qtbot.addWidget(w)
    assert not w.hasFocus()
    before = w.value()
    event = _wheel_event()
    w.wheelEvent(event)
    assert w.value() == before
    assert event.isAccepted() is False


def test_doublespinbox_ignores_wheel_when_unfocused(qtbot):
    w = NoWheelDoubleSpinBox()
    w.setRange(0.0, 10.0)
    w.setValue(2.5)
    qtbot.addWidget(w)
    assert not w.hasFocus()
    before = w.value()
    event = _wheel_event()
    w.wheelEvent(event)
    assert abs(w.value() - before) < 1e-9
    assert event.isAccepted() is False
