import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QCheckBox, QFileDialog
)

from llama_launcher.core.spec import Mount
from llama_launcher.ui.widgets.no_wheel import NoWheelComboBox
from llama_launcher.ui.widgets.table_columns import set_resizable_columns

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
        for _col, _tip in enumerate((
            "Folder on the host, e.g. /mnt/storage/AI/Models.",
            "Where it appears inside the container, e.g. /models. Reference model "
            "paths by this container path.",
            "Marks this mount's purpose (e.g. models) for the launcher's path mapping.",
            "Mount mode: ro (read-only) or rw (read-write).",
            "SELinux relabel flag (z/Z) for hosts that enforce SELinux.",
            "Set this mount as the container working directory.",
        )):
            item = self.table.horizontalHeaderItem(_col)
            if item is not None:
                item.setToolTip(_tip)
        # Mount paths routinely need more room, so every column is draggable.
        set_resizable_columns(self.table, (190, 130, 90, 60, 74, 66))
        # Floor the table so it can't be squeezed to ~1 row when the Environment
        # form is crowded -- keep 2-4 folder rows visible (header + ~4 rows).
        self.table.setMinimumHeight(148)
        layout.addWidget(self.table)
        row = QHBoxLayout()
        add = QPushButton("+ Add folder")
        rm = QPushButton("- Remove")
        add.clicked.connect(self._add_folder)
        rm.clicked.connect(self._remove_selected)
        row.addWidget(add); row.addWidget(rm)
        layout.addLayout(row)
        self.table.itemChanged.connect(lambda *_: self.changed.emit())

    def _add_row(self, m: Mount):
        prev = self.table.blockSignals(True)
        try:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(m.host))
            self.table.setItem(r, 1, QTableWidgetItem(m.container))
            role = NoWheelComboBox(); role.addItems(_ROLES); role.setCurrentText(m.role)
            mode = NoWheelComboBox(); mode.addItems(_MODES); mode.setCurrentText(m.mode)
            selinux = NoWheelComboBox(); selinux.addItems(_SELINUX)
            selinux.setCurrentText(m.selinux or "")
            workdir = QCheckBox(); workdir.setChecked(m.workdir)
            for w in (role, mode, selinux):
                w.currentTextChanged.connect(lambda *_: self.changed.emit())
            workdir.toggled.connect(lambda *_: self.changed.emit())
            self.table.setCellWidget(r, 2, role)
            self.table.setCellWidget(r, 3, mode)
            self.table.setCellWidget(r, 4, selinux)
            self.table.setCellWidget(r, 5, workdir)
        finally:
            self.table.blockSignals(prev)
        if not prev:
            self.changed.emit()

    def _add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Add folder")
        if not d:
            return
        self._add_row(Mount(host=d, container="/" + os.path.basename(d.rstrip("/")),
                            role="custom", mode="ro"))

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        self.changed.emit()

    def set_mounts(self, mounts: list[Mount]):
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(0)
            for m in mounts:
                self._add_row(m)
        finally:
            self.table.blockSignals(False)
        self.changed.emit()

    def mounts(self) -> list[Mount]:
        out = []
        for r in range(self.table.rowCount()):
            if self.table.cellWidget(r, 2) is None:
                continue  # row still mid-construction
            out.append(Mount(
                host=self.table.item(r, 0).text() if self.table.item(r, 0) else "",
                container=self.table.item(r, 1).text() if self.table.item(r, 1) else "",
                role=self.table.cellWidget(r, 2).currentText(),
                mode=self.table.cellWidget(r, 3).currentText(),
                selinux=self.table.cellWidget(r, 4).currentText() or None,
                workdir=self.table.cellWidget(r, 5).isChecked(),
            ))
        return out
