from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QToolButton, QToolTip


class InfoButton(QToolButton):
    """Flat 'ⓘ' button; click reveals help text in a popover, hover in a tooltip."""

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.info_text = text
        self.setText("ⓘ")
        self.setAutoRaise(True)
        self.setToolTip(text)
        self.clicked.connect(self._show)

    def _show(self) -> None:
        QToolTip.showText(QCursor.pos(), self.info_text, self)
