from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QDoubleSpinBox
)

from llama_launcher.core.spec import LoraRef


class LoraPanel(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Path", "Scale"])
        layout.addWidget(self.table)
        row = QHBoxLayout()
        add = QPushButton("+ Add")
        rm = QPushButton("- Remove")
        add.clicked.connect(self._add_blank)
        rm.clicked.connect(self._remove_selected)
        row.addWidget(add); row.addWidget(rm)
        layout.addLayout(row)
        self.table.itemChanged.connect(lambda *_: self.changed.emit())

    def _add_row(self, lora: LoraRef):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(lora.path))
        scale = QDoubleSpinBox()
        scale.setRange(0.0, 10.0)
        scale.setSingleStep(0.1)
        scale.setDecimals(2)
        scale.setValue(lora.scale)
        scale.valueChanged.connect(self.changed.emit)
        self.table.setCellWidget(r, 1, scale)
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
            out.append(LoraRef(
                path=path,
                scale=self.table.cellWidget(r, 1).value(),
            ))
        return out
