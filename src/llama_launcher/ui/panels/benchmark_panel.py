from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from llama_launcher.core.sweep import (
    MeasuredMemory,
    SweepPoint,
    best_point,
    largest_row,
    measured_card_total,
    measured_ram_total,
)
from llama_launcher.ui.widgets.info_button import InfoButton

_BENCH_TABLE_HEADERS = ["size", "prompt_n", "pp t/s", "gen t/s", "total s"]

# Per-column explanations shown as header tooltips; the InfoButton popover next
# to the table repeats the throughput ones inline so they're available without
# hovering, without keeping the reminder text always on screen.
_BENCH_HEADER_TIPS = {
    "size": "Target prompt length in tokens (a filler prompt is padded to this).",
    "prompt_n": "Actual number of prompt tokens sent to the server for this row.",
    "pp t/s": "Prefill (prompt-processing) throughput: tokens/sec the model "
    "ingests the prompt.",
    "gen t/s": "Generation throughput: tokens/sec the model produces in the reply.",
    "total s": "Total wall-clock seconds for this prompt size (prefill + generation).",
}

_BENCH_INTRO = (
    "Benchmark the running server: POSTs filler prompts and reads "
    "llama.cpp timings for prompt-eval / generation tok/s. History is "
    "kept per profile so you can A/B a flag or model change."
)

_BENCH_LEGEND = (
    "pp t/s = prefill (prompt-processing) throughput  \u00b7  "
    "gen t/s = generation throughput  \u00b7  total s = wall-clock per size"
)

_SWEEP_TABLE_HEADERS = ["count", "status", "ready s", "pp tok/s", "gen tok/s"]

_GIB = 1024**3

_SWEEP_INTRO = (
    "A sweep restarts the server once per count of the offload knob, from "
    "the smallest fitting count up to a chosen stop, and benchmarks each "
    "one. The table below lists every point tried, with the fastest ok "
    "point marked as best. Apply writes that count into the Configure form "
    "and saves the profile."
)


def _fmt_gib(nbytes) -> str:
    """One-decimal GiB value for a byte count."""
    return f"{nbytes / _GIB:.1f}"


class BenchmarkPanel(QWidget):
    """Speed benchmark for the running server: its own tab.

    Results are grouped per run: a header row labelled with the model and flags
    that produced it sits above that run's metric rows, so past runs stay tied to
    their model for A/B comparison (up to the store's cap). Clear wipes the
    on-disk history for the current profile.
    """

    benchmark_run_requested = Signal(dict)
    benchmark_cancel_requested = Signal()
    benchmark_clear_requested = Signal()
    sweep_run_requested = Signal(dict)
    sweep_cancel_requested = Signal()
    sweep_apply_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        bench_config = QHBoxLayout()
        bench_config.addWidget(InfoButton(_BENCH_INTRO))
        bench_config.addWidget(QLabel("Prompt sizes:"))
        self.bench_sizes = QLineEdit("128, 512, 2048")
        bench_config.addWidget(self.bench_sizes)
        bench_config.addWidget(QLabel("n-predict:"))
        self.bench_npredict = QSpinBox()
        self.bench_npredict.setRange(1, 1_000_000)
        self.bench_npredict.setValue(128)
        bench_config.addWidget(self.bench_npredict)
        bench_config.addWidget(QLabel("warmup:"))
        self.bench_warmup = QSpinBox()
        self.bench_warmup.setRange(0, 100)
        self.bench_warmup.setValue(1)
        bench_config.addWidget(self.bench_warmup)
        bench_config.addWidget(QLabel("repeats:"))
        self.bench_repeats = QSpinBox()
        self.bench_repeats.setRange(1, 100)
        self.bench_repeats.setValue(3)
        bench_config.addWidget(self.bench_repeats)
        self.bench_run_btn = QPushButton("Run")
        self.bench_run_btn.setEnabled(False)
        self.bench_run_btn.clicked.connect(self._on_bench_run_clicked)
        bench_config.addWidget(self.bench_run_btn)
        self.bench_clear_btn = QPushButton("Clear")
        self.bench_clear_btn.setToolTip(
            "Delete the saved benchmark history for this profile."
        )
        self.bench_clear_btn.clicked.connect(self.benchmark_clear_requested)
        bench_config.addWidget(self.bench_clear_btn)
        layout.addLayout(bench_config)

        progress_row = QHBoxLayout()
        self.bench_progress = QLabel("")
        progress_row.addWidget(self.bench_progress, 1)
        progress_row.addWidget(InfoButton(_BENCH_LEGEND))
        layout.addLayout(progress_row)

        self.bench_table = QTableWidget(0, len(_BENCH_TABLE_HEADERS))
        self.bench_table.setHorizontalHeaderLabels(_BENCH_TABLE_HEADERS)
        for c, name in enumerate(_BENCH_TABLE_HEADERS):
            self.bench_table.horizontalHeaderItem(c).setToolTip(
                _BENCH_HEADER_TIPS[name]
            )
        self.bench_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        sweep_row = QHBoxLayout()
        sweep_row.addWidget(InfoButton(_SWEEP_INTRO))
        self.sweep_knob_label = QLabel("Sweep --n-cpu-ffn")
        sweep_row.addWidget(self.sweep_knob_label)
        sweep_row.addWidget(QLabel("from:"))
        self.sweep_from = QSpinBox()
        self.sweep_from.setRange(0, 999)
        sweep_row.addWidget(self.sweep_from)
        sweep_row.addWidget(QLabel("to:"))
        self.sweep_to = QSpinBox()
        self.sweep_to.setRange(0, 999)
        self.sweep_to.setValue(8)
        sweep_row.addWidget(self.sweep_to)
        sweep_row.addWidget(QLabel("step:"))
        self.sweep_step = QSpinBox()
        self.sweep_step.setRange(1, 64)
        self.sweep_step.setValue(2)
        sweep_row.addWidget(self.sweep_step)
        sweep_row.addWidget(QLabel("ready timeout s:"))
        self.sweep_timeout = QSpinBox()
        self.sweep_timeout.setRange(30, 3600)
        self.sweep_timeout.setValue(600)
        sweep_row.addWidget(self.sweep_timeout)
        self.sweep_run_btn = QPushButton("Run sweep")
        self.sweep_run_btn.setEnabled(False)
        self.sweep_run_btn.clicked.connect(self._on_sweep_run_clicked)
        sweep_row.addWidget(self.sweep_run_btn)
        self.sweep_apply_btn = QPushButton("Apply")
        self.sweep_apply_btn.setEnabled(False)
        self.sweep_apply_btn.clicked.connect(self._on_sweep_apply_clicked)
        sweep_row.addWidget(self.sweep_apply_btn)
        layout.addLayout(sweep_row)
        self.sweep_status = QLabel("")
        self.sweep_status.setWordWrap(True)
        layout.addWidget(self.sweep_status)

        self.sweep_table = QTableWidget(0, len(_SWEEP_TABLE_HEADERS))
        self.sweep_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.sweep_table.horizontalHeader().setStretchLastSection(True)
        self._rebuild_sweep_headers(0)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.bench_table)
        splitter.addWidget(self.sweep_table)
        layout.addWidget(splitter, 1)

        self._bench_running = False
        self._sweep_running = False
        self._sweep_n_cards = 0
        self._sweep_best_count = None
        self._sweep_points = []

    def _on_bench_run_clicked(self) -> None:
        if self._bench_running:
            self.benchmark_cancel_requested.emit()
            return
        try:
            sizes = [int(s) for s in self.bench_sizes.text().split(",") if s.strip()]
        except ValueError:
            return
        self.benchmark_run_requested.emit(
            {
                "sizes": sizes,
                "n_predict": self.bench_npredict.value(),
                "warmup": self.bench_warmup.value(),
                "repeats": self.bench_repeats.value(),
            }
        )

    def _on_sweep_run_clicked(self) -> None:
        if self._sweep_running:
            self.sweep_cancel_requested.emit()
            return
        self.sweep_run_requested.emit(self.sweep_config())

    def _on_sweep_apply_clicked(self) -> None:
        if self._sweep_best_count is not None:
            self.sweep_apply_requested.emit(self._sweep_best_count)

    @staticmethod
    def _snapshot_label(snapshot: dict) -> str:
        """Compact join of non-null snapshot flags, e.g. '-ngl99 fa=on'."""
        parts = []
        for key, val in snapshot.items():
            if val is None:
                continue
            if key == "ngl":
                parts.append(f"-ngl{val}")
            else:
                parts.append(f"{key}={val}")
        return " ".join(parts)

    def set_benchmark_available(self, available: bool) -> None:
        self.bench_run_btn.setEnabled(available)

    def set_benchmark_running(self, running: bool) -> None:
        self._bench_running = running
        self.bench_run_btn.setText("Cancel" if running else "Run")
        for w in (
            self.bench_sizes,
            self.bench_npredict,
            self.bench_warmup,
            self.bench_repeats,
        ):
            w.setEnabled(not running)

    def set_benchmark_progress(self, text: str) -> None:
        self.bench_progress.setText(text)

    def show_benchmark_run(self, run: dict, delta: dict | None) -> None:
        """Append the delta summary vs the previous run to the progress line.

        The results table itself is (re)painted by set_benchmark_history, which
        MainWindow calls right after this with the full per-profile history.
        """
        if not delta:
            return
        # "shared" is a list of {"size","pp_pct","gen_pct"} -- one entry per
        # target size present in both runs (see benchmark_store.delta).
        parts = []
        for entry in delta.get("shared") or []:
            bits = []
            pp = entry.get("pp_pct")
            gen = entry.get("gen_pct")
            if pp is not None:
                bits.append(f"pp {pp:+.0f}%")
            if gen is not None:
                bits.append(f"gen {gen:+.0f}%")
            if bits:
                parts.append(f"{entry.get('size')}: " + " ".join(bits))
        summary = "\u0394 " + " \u00b7 ".join(parts) if parts else "\u0394"
        if delta.get("sizes_differ"):
            summary += " (sizes differ)"
        current = self.bench_progress.text()
        self.bench_progress.setText(f"{current}  {summary}" if current else summary)

    def _add_group_header(self, run: dict) -> None:
        """Insert a spanned, styled header row identifying a run's model/config."""
        row = self.bench_table.rowCount()
        self.bench_table.insertRow(row)
        label = self._snapshot_label(run.get("snapshot") or {})
        ts = run.get("timestamp", "")
        text = f"{ts}  \u00b7  {label}" if label else str(ts)
        item = QTableWidgetItem(text)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        item.setBackground(QBrush(QColor(0, 0, 0, 30)))
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)  # not selectable/editable
        self.bench_table.setItem(row, 0, item)
        self.bench_table.setSpan(row, 0, 1, len(_BENCH_TABLE_HEADERS))

    def _add_metric_rows(self, run: dict) -> None:
        for row in run.get("rows", []):
            r = self.bench_table.rowCount()
            self.bench_table.insertRow(r)
            values = [
                row.get("target_size"),
                row.get("prompt_n"),
                row.get("pp_tok_s"),
                row.get("gen_tok_s"),
                row.get("total_s"),
            ]
            for c, val in enumerate(values):
                self.bench_table.setItem(
                    r, c, QTableWidgetItem("" if val is None else str(val))
                )

    def set_benchmark_history(self, runs: list) -> None:
        """Repaint the table as one labelled group per stored run, newest first."""
        self.bench_table.clearSpans()
        self.bench_table.setRowCount(0)
        for run in sorted(runs, key=lambda r: r.get("timestamp") or "", reverse=True):
            self._add_group_header(run)
            self._add_metric_rows(run)

    def set_sweep_prefill(self, knob: str, start: int, stop: int, step: int) -> None:
        self.sweep_knob_label.setText(f"Sweep --{knob}")
        self.sweep_from.setValue(start)
        self.sweep_to.setValue(stop)
        self.sweep_step.setValue(step)

    def set_sweep_available(self, available: bool, reason: str = "") -> None:
        self.sweep_run_btn.setEnabled(available or self._sweep_running)
        if reason or not available:
            self.sweep_status.setText(reason)

    def set_sweep_running(self, running: bool) -> None:
        self._sweep_running = running
        self.sweep_run_btn.setText("Cancel sweep" if running else "Run sweep")
        for w in (self.sweep_from, self.sweep_to, self.sweep_step, self.sweep_timeout):
            w.setEnabled(not running)
        if running:
            self.sweep_apply_btn.setEnabled(False)
        else:
            self.sweep_apply_btn.setEnabled(self._sweep_best_count is not None)

    def sweep_config(self) -> dict:
        return {
            "start": self.sweep_from.value(),
            "stop": self.sweep_to.value(),
            "step": self.sweep_step.value(),
            "ready_timeout": self.sweep_timeout.value(),
        }

    def _rebuild_sweep_headers(self, n_cards: int) -> None:
        """Base columns plus one meas/est GiB column per card and a final RAM one."""
        headers = (
            list(_SWEEP_TABLE_HEADERS)
            + [f"GPU{i} meas / est GiB" for i in range(n_cards)]
            + ["RAM meas / est GiB"]
        )
        self.sweep_table.setColumnCount(len(headers))
        self.sweep_table.setHorizontalHeaderLabels(headers)
        # The RAM column's label is the widest header and gets clipped at the
        # default column width, so the last section always stretches to fit,
        # even if a rebuild replaces the header.
        self.sweep_table.horizontalHeader().setStretchLastSection(True)

    @staticmethod
    def _to_sweep_point(pt: dict) -> SweepPoint:
        """Build a SweepPoint from a possibly partial dict (an older or
        truncated store file, or a point still running), never raising."""
        measured = pt.get("measured")
        mem = None
        if measured and "cards" in measured and "ram" in measured:
            mem = MeasuredMemory(cards=tuple(measured["cards"]), ram=measured["ram"])
        return SweepPoint(
            count=pt.get("count", 0),
            status=pt.get("status", ""),
            ready_seconds=pt.get("ready_seconds"),
            rows=tuple(pt.get("rows") or []),
            measured=mem,
            estimated_cards=tuple(pt.get("estimated_cards") or []),
            estimated_ram=pt.get("estimated_ram"),
            error=pt.get("error", ""),
        )

    def _add_sweep_row(self, pt: SweepPoint, is_best: bool) -> None:
        row = self.sweep_table.rowCount()
        self.sweep_table.insertRow(row)
        count_text = f"{pt.count} (best)" if is_best else str(pt.count)
        self.sweep_table.setItem(row, 0, QTableWidgetItem(count_text))
        status_item = QTableWidgetItem(pt.status)
        if pt.error:
            status_item.setToolTip(pt.error)
        self.sweep_table.setItem(row, 1, status_item)
        ready = "-" if pt.ready_seconds is None else f"{pt.ready_seconds:.1f}"
        self.sweep_table.setItem(row, 2, QTableWidgetItem(ready))
        largest = largest_row(pt.rows)
        if largest is not None:
            pp_val = largest.get("pp_tok_s")
            gen_val = largest.get("gen_tok_s")
            pp = "-" if pp_val is None else f"{pp_val:.1f}"
            gen = "-" if gen_val is None else f"{gen_val:.1f}"
        else:
            pp = gen = "-"
        self.sweep_table.setItem(row, 3, QTableWidgetItem(pp))
        self.sweep_table.setItem(row, 4, QTableWidgetItem(gen))
        col = 5
        for i in range(self._sweep_n_cards):
            if pt.measured is None or i >= len(pt.measured.cards):
                meas = "-"
            else:
                meas = _fmt_gib(measured_card_total(pt.measured, i))
            est = (
                "-" if i >= len(pt.estimated_cards) else _fmt_gib(pt.estimated_cards[i])
            )
            self.sweep_table.setItem(row, col, QTableWidgetItem(f"{meas} / {est}"))
            col += 1
        ram_meas = (
            "-" if pt.measured is None else _fmt_gib(measured_ram_total(pt.measured))
        )
        ram_est = "-" if pt.estimated_ram is None else _fmt_gib(pt.estimated_ram)
        self.sweep_table.setItem(row, col, QTableWidgetItem(f"{ram_meas} / {ram_est}"))
        if is_best:
            highlight = QBrush(QColor(0, 0, 0, 30))
            for c in range(self.sweep_table.columnCount()):
                self.sweep_table.item(row, c).setBackground(highlight)

    def _render_sweep_points(self, points: list) -> None:
        self._sweep_points = points
        best = best_point(points)
        self._sweep_best_count = best.count if best else None
        self.sweep_table.setRowCount(0)
        for pt in points:
            self._add_sweep_row(pt, is_best=best is not None and pt is best)
        self.sweep_apply_btn.setEnabled(best is not None and not self._sweep_running)

    def show_sweep(self, sweep: dict, n_cards: int) -> None:
        """Repaint the sweep table for one sweep, header rebuilt for n_cards."""
        self._sweep_n_cards = n_cards
        self._rebuild_sweep_headers(n_cards)
        points = [self._to_sweep_point(p) for p in sweep.get("points", [])]
        self._render_sweep_points(points)

    def set_sweep_point(self, point: dict) -> None:
        """Update the row for this count if seen already, else append it."""
        pt = self._to_sweep_point(point)
        points = list(self._sweep_points)
        for i, existing in enumerate(points):
            if existing.count == pt.count:
                points[i] = pt
                break
        else:
            points.append(pt)
        self._render_sweep_points(points)

    def clear_sweep(self) -> None:
        """Empty the sweep table and the state read from it: no points, no
        best count and Apply disabled."""
        self._sweep_points = []
        self._sweep_best_count = None
        self.sweep_table.setRowCount(0)
        self.sweep_apply_btn.setEnabled(False)

    def reset(self):
        self.bench_table.clearSpans()
        self.bench_table.setRowCount(0)
        self.bench_progress.setText("")
        self.clear_sweep()
