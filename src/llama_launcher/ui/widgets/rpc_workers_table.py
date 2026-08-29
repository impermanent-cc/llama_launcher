from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
)

from llama_launcher.core.spec import RpcWorker
from llama_launcher.ui.widgets.no_wheel import NoWheelComboBox, NoWheelSpinBox
from llama_launcher.ui.widgets.table_columns import set_resizable_columns

_DEVICES = ["CPU", "CUDA0", "CUDA1"]


class RpcWorkersTable(QWidget):
    """RPC-pool launch mode: one row per worker (Node · Device · Mem MB · Port).

    Mirrors MountsPanel/LoraPanel's table-of-cell-widgets pattern (see
    ui/panels/mounts_panel.py) rather than editable QTableWidgetItems, since
    every column here is an enum/number best picked from a combo/spin box.
    """

    changed = Signal()

    def __init__(self, node_names: list[str] | None = None, parent=None):
        super().__init__(parent)
        self._node_names = list(node_names or [])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Node", "Device", "Contribution MB", "Port"])
        # Current llama.cpp `ggml-rpc-server` has no per-worker memory-cap flag,
        # so this value is NOT enforced on the worker; it is only a pledge feeding
        # the "Check fit" preflight (pooled VRAM+RAM vs. model size).
        self.table.horizontalHeaderItem(2).setToolTip(
            "Estimate only: how much memory you expect this worker to donate, "
            "used by 'Check fit'. Not enforced (rpc-server has no memory cap).")
        set_resizable_columns(self.table, (140, 90, 130, 70))
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        add = QPushButton("+ Add worker")
        rm = QPushButton("- Remove")
        add.clicked.connect(self._add_blank)
        rm.clicked.connect(self._remove_selected)
        row.addWidget(add)
        row.addWidget(rm)
        layout.addLayout(row)

    def set_node_names(self, node_names: list[str]) -> None:
        """Refresh the choices offered by each row's Node combo (e.g. after
        the Nodes dialog adds/removes a registered node), keeping each row's
        current selection if it still exists."""
        self._node_names = list(node_names)
        for r in range(self.table.rowCount()):
            combo = self.table.cellWidget(r, 0)
            if combo is None:
                continue
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(self._node_names)
            idx = combo.findText(cur)
            if idx < 0 and cur:
                combo.addItem(cur)
                idx = combo.findText(cur)
            combo.setCurrentIndex(max(0, idx))
            combo.blockSignals(False)

    def _add_row(self, w: RpcWorker) -> None:
        prev = self.table.blockSignals(True)
        try:
            r = self.table.rowCount()
            self.table.insertRow(r)

            node = NoWheelComboBox()
            node.addItems(self._node_names)
            idx = node.findText(w.node)
            if idx < 0 and w.node:
                node.addItem(w.node)
                idx = node.findText(w.node)
            node.setCurrentIndex(max(0, idx))
            node.currentTextChanged.connect(lambda *_: self.changed.emit())
            self.table.setCellWidget(r, 0, node)

            device = NoWheelComboBox()
            device.addItems(_DEVICES)
            didx = device.findText(w.device)
            if didx < 0:
                device.addItem(w.device)
                didx = device.findText(w.device)
            device.setCurrentIndex(max(0, didx))
            device.currentTextChanged.connect(lambda *_: self.changed.emit())
            self.table.setCellWidget(r, 1, device)

            mem = NoWheelSpinBox()
            mem.setRange(0, 1_000_000)
            mem.setSuffix(" MB")
            mem.setValue(w.mem_mb)
            mem.valueChanged.connect(lambda *_: self.changed.emit())
            self.table.setCellWidget(r, 2, mem)

            port = NoWheelSpinBox()
            port.setRange(1, 65535)
            port.setValue(w.port)
            port.valueChanged.connect(lambda *_: self.changed.emit())
            self.table.setCellWidget(r, 3, port)
        finally:
            self.table.blockSignals(prev)
        if not prev:
            self.changed.emit()

    def _add_blank(self) -> None:
        default_node = self._node_names[0] if self._node_names else "local"
        self._add_row(RpcWorker(node=default_node))

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        self.changed.emit()

    def set_workers(self, workers: list[RpcWorker]) -> None:
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(0)
            for w in workers:
                self._add_row(w)
        finally:
            self.table.blockSignals(False)
        self.changed.emit()

    def workers(self) -> list[RpcWorker]:
        out = []
        for r in range(self.table.rowCount()):
            node_w = self.table.cellWidget(r, 0)
            if node_w is None:
                continue   # row still mid-construction
            out.append(RpcWorker(
                node=node_w.currentText(),
                device=self.table.cellWidget(r, 1).currentText(),
                mem_mb=self.table.cellWidget(r, 2).value(),
                port=self.table.cellWidget(r, 3).value(),
            ))
        return out
