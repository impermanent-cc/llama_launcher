from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QFileDialog
)

from llama_launcher.core.spec import LoraRef
from llama_launcher.ui.widgets.no_wheel import NoWheelDoubleSpinBox
from llama_launcher.ui.widgets.table_columns import set_resizable_columns


class LoraPanel(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._browse_resolver = None
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Path", "Scale", ""])
        # Path + Scale user-resizable; the fixed Browse-button column keeps
        # sizing itself to its content.
        set_resizable_columns(self.table, (240, 60), content_cols=(2,))
        layout.addWidget(self.table)
        row = QHBoxLayout()
        add = QPushButton("+ Add")
        rm = QPushButton("- Remove")
        add.clicked.connect(self._add_blank)
        rm.clicked.connect(self._remove_selected)
        row.addWidget(add); row.addWidget(rm)
        layout.addLayout(row)
        self.table.itemChanged.connect(lambda *_: self.changed.emit())

    def set_browse_resolver(self, fn):
        self._browse_resolver = fn

    def _resolve(self, host_path: str):
        return self._browse_resolver(host_path) if self._browse_resolver else host_path

    def _add_row(self, lora: LoraRef):
        prev = self.table.blockSignals(True)
        try:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(lora.path))
            scale = NoWheelDoubleSpinBox()
            scale.setRange(0.0, 10.0)
            scale.setSingleStep(0.1)
            scale.setDecimals(2)
            scale.setValue(lora.scale)
            scale.valueChanged.connect(lambda *_: self.changed.emit())
            self.table.setCellWidget(r, 1, scale)
            browse_btn = QPushButton("Browse…")
            row_index = r

            def _make_browse(row):
                def _browse():
                    path, _ = QFileDialog.getOpenFileName(self, "Select LoRA", "")
                    if not path:
                        return
                    resolved = self._resolve(path)
                    if resolved is None:
                        return
                    item = self.table.item(row, 0)
                    if item is not None:
                        item.setText(resolved)
                return _browse

            browse_btn.clicked.connect(_make_browse(row_index))
            self.table.setCellWidget(r, 2, browse_btn)
        finally:
            self.table.blockSignals(prev)
        if not prev:
            self.changed.emit()

    def _add_blank(self):
        self._add_row(LoraRef(path="", scale=1.0))

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        self.changed.emit()

    def set_loras(self, loras: list[LoraRef]):
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(0)
            for lora in loras:
                self._add_row(lora)
        finally:
            self.table.blockSignals(False)
        self.changed.emit()

    def loras(self) -> list[LoraRef]:
        out = []
        for r in range(self.table.rowCount()):
            path = self.table.item(r, 0).text() if self.table.item(r, 0) else ""
            if not path:
                continue
            if self.table.cellWidget(r, 1) is None:
                continue  # row still mid-construction
            out.append(LoraRef(
                path=path,
                scale=self.table.cellWidget(r, 1).value(),
            ))
        return out
