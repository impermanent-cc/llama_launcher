from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox


class _NoWheelMixin:
    """Mixin that stops the mouse wheel from changing the value on hover.

    - Sets StrongFocus so the widget only takes focus on click/tab (not hover).
    - wheelEvent is ignored (so an enclosing QScrollArea scrolls instead),
      unless the widget already has focus, in which case the normal behaviour
      (changing the value) applies.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class NoWheelComboBox(_NoWheelMixin, QComboBox):
    pass


class NoWheelSpinBox(_NoWheelMixin, QSpinBox):
    pass


class NoWheelDoubleSpinBox(_NoWheelMixin, QDoubleSpinBox):
    pass
