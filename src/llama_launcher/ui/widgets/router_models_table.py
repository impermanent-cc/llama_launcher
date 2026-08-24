from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QHeaderView, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from llama_launcher.ui.widgets.info_button import InfoButton


def _status_text(model) -> str:
    if model.failed:
        code = "" if model.exit_code is None else f" (exit {model.exit_code})"
        return f"failed{code}"
    if model.progress is not None and model.status in ("loading", "downloading"):
        return f"{model.status} {model.progress * 100:.0f}%"
    return model.status


class RouterModelsTable(QWidget):
    """Model id / status / per-row load-unload. Capped to ~3-4 rows, then scrolls."""

    load_requested = Signal(str)
    unload_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._load_buttons: dict = {}
        self._unload_buttons: dict = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(QLabel("Models"))
        self.info = InfoButton(
            "Model ids the router exposes: each row's Model id is what API "
            "clients request. Load or unload a member model here; Status shows "
            "whether it's currently up."
        )
        header.addWidget(self.info)
        header.addStretch(1)
        root.addLayout(header)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Model id", "Status", ""])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMaximumHeight(140)   # ~3-4 rows, own scrollbar past that
        root.addWidget(self.table)

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
