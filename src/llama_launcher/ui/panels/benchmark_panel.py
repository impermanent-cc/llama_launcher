from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QSpinBox, QTableWidget, QTableWidgetItem, QListWidget,
    QAbstractItemView,
)

_BENCH_TABLE_HEADERS = ["size", "prompt_n", "pp t/s", "gen t/s", "total s"]


class BenchmarkPanel(QWidget):
    """Speed benchmark for the running server: its own tab.

    Split out of MonitorPanel so the Monitor tab is just instances + logs; the
    controls, results table and per-profile history live here. Wiring in
    MainWindow (run/cancel handlers, availability, history) is unchanged -- only
    the widget these methods live on moved.
    """
    benchmark_run_requested = Signal(dict)
    benchmark_cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "Benchmark the running server: POSTs filler prompts and reads "
            "llama.cpp timings for prompt-eval / generation tok/s. History is "
            "kept per profile so you can A/B a flag or model change."))

        bench_config = QHBoxLayout()
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
        layout.addLayout(bench_config)

        self.bench_progress = QLabel("")
        layout.addWidget(self.bench_progress)

        self.bench_table = QTableWidget(0, len(_BENCH_TABLE_HEADERS))
        self.bench_table.setHorizontalHeaderLabels(_BENCH_TABLE_HEADERS)
        self.bench_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.bench_table, 1)

        layout.addWidget(QLabel("History (most recent first):"))
        self.bench_history = QListWidget()
        self.bench_history.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.bench_history, 1)

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
        rows = run.get("rows", [])
        self.bench_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            values = [row.get("target_size"), row.get("prompt_n"), row.get("pp_tok_s"),
                      row.get("gen_tok_s"), row.get("total_s")]
            for c, val in enumerate(values):
                self.bench_table.setItem(r, c, QTableWidgetItem("" if val is None else str(val)))
        if delta:
            # "shared" is a list of {"size","pp_pct","gen_pct"} -- one entry
            # per target size present in both runs (see benchmark_store.delta).
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
            summary = "Δ " + " · ".join(parts) if parts else "Δ"
            if delta.get("sizes_differ"):
                summary += " (sizes differ)"
            current = self.bench_progress.text()
            self.bench_progress.setText(f"{current}  {summary}" if current else summary)

    def set_benchmark_history(self, runs: list) -> None:
        self.bench_history.clear()
        for run in sorted(runs, key=lambda r: r.get("timestamp") or "", reverse=True):
            label = self._snapshot_label(run.get("snapshot") or {})
            ts = run.get("timestamp", "")
            self.bench_history.addItem(f"{ts}  {label}" if label else str(ts))

    def reset(self):
        self.bench_table.setRowCount(0)
        self.bench_history.clear()
        self.bench_progress.setText("")
