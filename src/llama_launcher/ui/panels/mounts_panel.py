from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QComboBox, QCheckBox
)

from llama_launcher.core.spec import Mount

_ROLES = ["model", "workspace", "custom"]
_MODES = ["ro", "rw"]
_SELINUX = ["", "z", "Z"]


class MountsPanel(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Host", "Container", "Role", "Mode", "SELinux", "Workdir"])
        layout.addWidget(self.table)
        row = QHBoxLayout()
        add = QPushButton("+ Add folder")
        rm = QPushButton("- Remove")
        add.clicked.connect(self._add_blank)
        rm.clicked.connect(self._remove_selected)
        row.addWidget(add); row.addWidget(rm)
        layout.addLayout(row)

    def _add_row(self, m: Mount):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(m.host))
        self.table.setItem(r, 1, QTableWidgetItem(m.container))
        role = QComboBox(); role.addItems(_ROLES); role.setCurrentText(m.role)
        mode = QComboBox(); mode.addItems(_MODES); mode.setCurrentText(m.mode)
        selinux = QComboBox(); selinux.addItems(_SELINUX)
        selinux.setCurrentText(m.selinux or "")
        workdir = QCheckBox(); workdir.setChecked(m.workdir)
        for w in (role, mode, selinux):
            w.currentTextChanged.connect(self.changed.emit)
        workdir.toggled.connect(self.changed.emit)
        self.table.setCellWidget(r, 2, role)
        self.table.setCellWidget(r, 3, mode)
        self.table.setCellWidget(r, 4, selinux)
        self.table.setCellWidget(r, 5, workdir)
        self.changed.emit()

    def _add_blank(self):
        self._add_row(Mount(host="", container="", role="custom", mode="ro"))

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        self.changed.emit()

    def set_mounts(self, mounts: list[Mount]):
        self.table.setRowCount(0)
        for m in mounts:
            self._add_row(m)

    def mounts(self) -> list[Mount]:
        out = []
        for r in range(self.table.rowCount()):
            out.append(Mount(
                host=self.table.item(r, 0).text() if self.table.item(r, 0) else "",
                container=self.table.item(r, 1).text() if self.table.item(r, 1) else "",
                role=self.table.cellWidget(r, 2).currentText(),
                mode=self.table.cellWidget(r, 3).currentText(),
                selinux=self.table.cellWidget(r, 4).currentText() or None,
                workdir=self.table.cellWidget(r, 5).isChecked(),
            ))
        return out
