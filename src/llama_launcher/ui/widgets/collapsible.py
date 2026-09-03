from PySide6.QtCore import Qt
from PySide6.QtWidgets import QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    def __init__(
        self, title: str, content: QWidget, collapsed: bool = True, parent=None
    ):
        super().__init__(parent)
        self._content = content

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._button = QToolButton()
        self._button.setText(title)
        self._button.setCheckable(True)
        self._button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._button.toggled.connect(self._on_toggled)
        layout.addWidget(self._button)
        layout.addWidget(self._content)

        self.set_expanded(not collapsed)

    def _on_toggled(self, checked: bool):
        self._content.setVisible(checked)
        self._button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    def is_expanded(self) -> bool:
        return self._button.isChecked()

    def set_expanded(self, expanded: bool):
        self._button.setChecked(expanded)
        # ensure visuals are correct even if checked state didn't change
        self._on_toggled(expanded)
