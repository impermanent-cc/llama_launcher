from PySide6.QtCore import QRunnable, QThreadPool, QTimer, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from llama_launcher.core.spec import LoraRef
from llama_launcher.services import lora_api
from llama_launcher.ui.widgets.no_wheel import NoWheelDoubleSpinBox
from llama_launcher.ui.widgets.table_columns import set_resizable_columns


class _LoraCall(QRunnable):
    """Off-UI-thread wrapper for one blocking lora_api call.

    Applying scales is a POST with a 10s timeout, so it cannot run on the UI
    thread. Same done-flag poll shape as build_panel's gatherers.
    """

    def __init__(self, fn):
        super().__init__()
        self._fn = fn
        self.done = False
        self.result = None

    def run(self):
        try:
            self.result = self._fn()
        except Exception:  # worker must never raise
            self.result = None
        finally:
            self.done = True


class LoraPanel(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._browse_resolver = None
        self._live_resolver = None
        self._call = None
        self._poll = None
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
        row.addWidget(add)
        row.addWidget(rm)
        layout.addLayout(row)

        # Live control. Scales are the one part of a LoRA setup a running server
        # will change without a reload, so these two buttons are the difference
        # between "restart to try 0.5" and "try 0.5". They are enabled whenever
        # the profile could have a server (i.e. not a router, whose members hold
        # their own adapters); whether one is actually UP is answered by the
        # request itself, reported in the status line below. Deliberately not
        # gated on container_state(): that shells out to podman, and this check
        # would run on every edit to the form.
        live = QHBoxLayout()
        self.sync_btn = QPushButton("Sync from server")
        self.apply_btn = QPushButton("Apply to server")
        self.sync_btn.setToolTip(
            "Read the scales the RUNNING server is using and load them into the "
            "table above."
        )
        self.apply_btn.setToolTip(
            "Push the scales above to the running server, no restart. Paths must "
            "match the adapters it was launched with."
        )
        self.sync_btn.clicked.connect(self.sync_from_server)
        self.apply_btn.clicked.connect(self.apply_to_server)
        live.addWidget(self.sync_btn)
        live.addWidget(self.apply_btn)
        layout.addLayout(live)
        self.live_status = QLabel("")
        self.live_status.setWordWrap(True)
        layout.addWidget(self.live_status)

        self.table.itemChanged.connect(lambda *_: self.changed.emit())
        self.refresh_live_enabled()

    def set_browse_resolver(self, fn):
        self._browse_resolver = fn

    def set_live_resolver(self, fn):
        """`fn() -> (host, port, api_key) | None` for the profile's running server.

        None means "not running / not reachable", which is what disables the
        live buttons.
        """
        self._live_resolver = fn
        self.refresh_live_enabled()

    def _live_target(self):
        if self._live_resolver is None:
            return None
        try:
            return self._live_resolver()
        except Exception:  # a resolver must not break the form
            return None

    def refresh_live_enabled(self):
        """Enable the live buttons only while a server is actually resolvable."""
        busy = self._call is not None
        live = self._live_target() is not None and not busy
        self.sync_btn.setEnabled(live)
        self.apply_btn.setEnabled(live)

    def _start(self, fn, on_done):
        """Run `fn` off-thread, then `on_done(result)` back on the UI thread."""
        if self._call is not None:
            return
        self._call = _LoraCall(fn)
        self.refresh_live_enabled()
        QThreadPool.globalInstance().start(self._call)
        self._poll = QTimer(self)
        self._poll.setInterval(100)

        def _tick():
            call = self._call
            if call is None or not call.done:
                return
            self._poll.stop()
            self._call = None
            self.refresh_live_enabled()
            on_done(call.result)

        self._poll.timeout.connect(_tick)
        self._poll.start()

    def sync_from_server(self):
        target = self._live_target()
        if target is None:
            return
        host, port, key = target
        self.live_status.setText("Reading adapters\u2026")
        self._start(
            lambda: lora_api.list_adapters(host, port, key, timeout=5.0),
            self._apply_synced,
        )

    def _apply_synced(self, adapters):
        if adapters is None:
            self.live_status.setText("Could not reach the server.")
            return
        if not adapters:
            self.live_status.setText(
                "Server is running but was launched with no LoRA adapters. "
                "Adapters are chosen at launch; add them here and restart."
            )
            return
        by_path = {a.path: a for a in adapters}
        matched = 0
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            widget = self.table.cellWidget(r, 1)
            if item is None or widget is None:
                continue
            found = by_path.get(item.text())
            if found is not None:
                widget.setValue(found.scale)
                matched += 1
        active = sum(1 for a in adapters if a.active)
        self.live_status.setText(
            f"Server has {len(adapters)} adapter(s), {active} active; "
            f"matched {matched} of the rows above by path."
        )

    def apply_to_server(self):
        target = self._live_target()
        if target is None:
            return
        host, port, key = target
        rows = {lora.path: lora.scale for lora in self.loras()}
        if not rows:
            self.live_status.setText("Nothing to apply: no adapter rows.")
            return
        self.live_status.setText("Applying scales\u2026")

        def _work():
            # Re-read first: ids are assigned by the server at load time, so the
            # table's paths have to be resolved to live ids every time rather
            # than cached across launches.
            adapters = lora_api.list_adapters(host, port, key, timeout=5.0)
            if adapters is None:
                return ("unreachable", 0)
            if not adapters:
                return ("none-loaded", 0)
            # Send EVERY loaded adapter, not just matched rows: a partial list is
            # ambiguous upstream. Unmatched adapters keep the scale the server
            # already reports, so this never silently zeroes one the form does
            # not happen to list.
            scales = {a.id: rows.get(a.path, a.scale) for a in adapters}
            matched = sum(1 for a in adapters if a.path in rows)
            ok = lora_api.set_scales(host, port, key, scales, timeout=10.0)
            return ("ok" if ok else "failed", matched)

        self._start(_work, self._apply_done)

    def _apply_done(self, result):
        if result is None:
            self.live_status.setText("Apply failed.")
            return
        state, matched = result
        if state == "unreachable":
            self.live_status.setText("Could not reach the server.")
        elif state == "none-loaded":
            self.live_status.setText(
                "Server has no LoRA adapters loaded; scales apply only to "
                "adapters passed at launch."
            )
        elif state == "failed":
            self.live_status.setText("Server rejected the scale change.")
        else:
            self.live_status.setText(f"Applied. {matched} adapter(s) rescaled live.")

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
            browse_btn = QPushButton("Browse\u2026")
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
            out.append(
                LoraRef(
                    path=path,
                    scale=self.table.cellWidget(r, 1).value(),
                )
            )
        return out
