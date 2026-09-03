from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QSpinBox, QTableWidget, QTableWidgetItem,
    QAbstractItemView,
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

_BENCH_INTRO = ("Benchmark the running server: POSTs filler prompts and reads "
                 "llama.cpp timings for prompt-eval / generation tok/s. History is "
                 "kept per profile so you can A/B a flag or model change.")

_BENCH_LEGEND = ("pp t/s = prefill (prompt-processing) throughput  \u00b7  "
                 "gen t/s = generation throughput  \u00b7  total s = wall-clock per size")


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
        self.bench_clear_btn.setToolTip("Delete the saved benchmark history for this profile.")
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
            self.bench_table.horizontalHeaderItem(c).setToolTip(_BENCH_HEADER_TIPS[name])
        self.bench_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.bench_table, 1)

        self._bench_running = False

    def _on_bench_run_clicked(self) -> None:
        if self._bench_running:
            self.benchmark_cancel_requested.emit()
            return
        try:
            sizes = [int(s) for s in self.bench_sizes.text().split(",") if s.strip()]
        except ValueError:
            return
        self.benchmark_run_requested.emit({
            "sizes": sizes,
            "n_predict": self.bench_npredict.value(),
            "warmup": self.bench_warmup.value(),
            "repeats": self.bench_repeats.value(),
        })

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
        for w in (self.bench_sizes, self.bench_npredict, self.bench_warmup, self.bench_repeats):
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
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)     # not selectable/editable
        self.bench_table.setItem(row, 0, item)
        self.bench_table.setSpan(row, 0, 1, len(_BENCH_TABLE_HEADERS))

    def _add_metric_rows(self, run: dict) -> None:
        for row in run.get("rows", []):
            r = self.bench_table.rowCount()
            self.bench_table.insertRow(r)
            values = [row.get("target_size"), row.get("prompt_n"), row.get("pp_tok_s"),
                      row.get("gen_tok_s"), row.get("total_s")]
            for c, val in enumerate(values):
                self.bench_table.setItem(r, c, QTableWidgetItem("" if val is None else str(val)))

    def set_benchmark_history(self, runs: list) -> None:
        """Repaint the table as one labelled group per stored run, newest first."""
        self.bench_table.clearSpans()
        self.bench_table.setRowCount(0)
        for run in sorted(runs, key=lambda r: r.get("timestamp") or "", reverse=True):
            self._add_group_header(run)
            self._add_metric_rows(run)

    def reset(self):
        self.bench_table.clearSpans()
        self.bench_table.setRowCount(0)
        self.bench_progress.setText("")
