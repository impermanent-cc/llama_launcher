from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class StatusBanner(QWidget):
    """Router status line + exposure-warning banner (security-critical)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.status_label = QLabel("disconnected")
        root.addWidget(self.status_label)
        self.banner = QLabel("")
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet("QLabel { color: #b35c00; font-weight: bold; }")
        self.banner.setVisible(False)
        root.addWidget(self.banner)

    def set_connected(self, connected: bool) -> None:
        self.status_label.setText("● connected" if connected else "● disconnected")

    def set_error(self, text: str) -> None:
        """Report a failed control-plane action next to the buttons that caused it."""
        self.status_label.setText(f"● {text}" if text else "● connected")

    def set_exposure_warning(self, text: str) -> None:
        self.banner.setText(text)
        self.banner.setVisible(bool(text))
