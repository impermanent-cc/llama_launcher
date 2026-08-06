from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QHBoxLayout, QHeaderView, QLabel, QPlainTextEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

_MASK = "••••••••••••"


def _status_text(model) -> str:
    if model.failed:
        code = "" if model.exit_code is None else f" (exit {model.exit_code})"
        return f"failed{code}"
    if model.progress is not None and model.status in ("loading", "downloading"):
        return f"{model.status} {model.progress * 100:.0f}%"
    return model.status


class RouterPanel(QWidget):
    """Router control plane: model list, per-row load/unload, harness setup."""

    load_requested = Signal(str)
    unload_requested = Signal(str)
    regenerate_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._api_key = ""
        self._base_url = ""
        self._revealed = False
        self._load_buttons: dict = {}
        self._unload_buttons: dict = {}

        root = QVBoxLayout(self)

        self.status_label = QLabel("disconnected")
        root.addWidget(self.status_label)

        self.banner = QLabel("")
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet("QLabel { color: #b35c00; font-weight: bold; }")
        self.banner.setVisible(False)
        root.addWidget(self.banner)

        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("API key:"))
        self.key_label = QLabel(_MASK)
        self.key_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        key_row.addWidget(self.key_label, 1)
        self.reveal_check = QCheckBox("Reveal")
        self.reveal_check.toggled.connect(self.reveal_key)
        key_row.addWidget(self.reveal_check)
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.clicked.connect(self._copy_key)
        key_row.addWidget(self.copy_btn)
        self.regen_btn = QPushButton("Regenerate")
        self.regen_btn.clicked.connect(self.regenerate_requested.emit)
        key_row.addWidget(self.regen_btn)
        root.addLayout(key_row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Model id", "Status", ""])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        root.addWidget(self.table, 1)

        root.addWidget(QLabel("Harness setup"))
        self.harness_text = QPlainTextEdit()
        self.harness_text.setReadOnly(True)
        self.harness_text.setMaximumHeight(120)
        root.addWidget(self.harness_text)

    # -- state ---------------------------------------------------------------

    def set_connected(self, connected: bool) -> None:
        self.status_label.setText("● connected" if connected else "● disconnected")

    def set_error(self, text: str) -> None:
        """Report a failed control-plane action next to the buttons that caused it."""
        self.status_label.setText(f"● {text}" if text else "● connected")

    def set_exposure_warning(self, text: str) -> None:
        self.banner.setText(text)
        self.banner.setVisible(bool(text))

    def reveal_key(self, revealed: bool) -> None:
        self._revealed = bool(revealed)
        self.key_label.setText(self._api_key if self._revealed and self._api_key else _MASK)

    def _copy_key(self) -> None:
        from PySide6.QtWidgets import QApplication
        if self._api_key:
            QApplication.clipboard().setText(self._api_key)

    def set_endpoint(self, base_url: str, api_key: str, model_ids: list) -> None:
        self._base_url = base_url
        self._api_key = api_key or ""
        self.reveal_key(self._revealed)
        ids = "\n".join(f"  - {m}" for m in model_ids) or "  (no members yet)"
        self.harness_text.setPlainText(
            f"base_url: {base_url}/v1\n"
            f"api_key:  (see above — reveal to copy)\n"
            f"model ids:\n{ids}\n"
            f"\nCold loads can take minutes: raise your harness's request and "
            f"stale timeouts accordingly."
        )

    def set_models(self, models: list) -> None:
        self._load_buttons.clear()
        self._unload_buttons.clear()
        self.table.setRowCount(len(models))
        for row, model in enumerate(models):
            self.table.setItem(row, 0, QTableWidgetItem(model.id))
            self.table.setItem(row, 1, QTableWidgetItem(_status_text(model)))

            holder = QWidget()
            box = QHBoxLayout(holder)
            box.setContentsMargins(0, 0, 0, 0)

            load_btn = QPushButton("Load")
            load_btn.clicked.connect(
                lambda _checked=False, mid=model.id: self.load_requested.emit(mid))
            unload_btn = QPushButton("Unload")
            unload_btn.clicked.connect(
                lambda _checked=False, mid=model.id: self.unload_requested.emit(mid))

            load_btn.setEnabled(model.status in ("unloaded", "sleeping"))
            unload_btn.setEnabled(model.status in ("loaded", "sleeping", "loading"))

            box.addWidget(load_btn)
            box.addWidget(unload_btn)
            self.table.setCellWidget(row, 2, holder)

            self._load_buttons[model.id] = load_btn
            self._unload_buttons[model.id] = unload_btn
