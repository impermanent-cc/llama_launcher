from collections import deque

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QLineEdit, QSpinBox, QTableWidget, QTableWidgetItem, QListWidget,
    QAbstractItemView,
)

from llama_launcher.core.mtp_stats import parse_draft_stats, sparkline

_BENCH_TABLE_HEADERS = ["size", "prompt_n", "pp t/s", "gen t/s", "total s"]


class MonitorPanel(QWidget):
    enable_metrics_requested = Signal()
    benchmark_run_requested = Signal(dict)
    benchmark_cancel_requested = Signal()
    instance_selected = Signal(str)
    instance_stop_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # Create instances table
        from PySide6.QtWidgets import QHeaderView
        self.instances_table = QTableWidget(0, 5)
        self.instances_table.setHorizontalHeaderLabels(["Profile", "Port", "Health", "Stat", ""])
        self.instances_table.verticalHeader().setVisible(False)
        self.instances_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.instances_table.cellClicked.connect(lambda r, _c: self._emit_selected_for_row(r))
        self._instance_names: list[str] = []
        layout.insertWidget(0, self.instances_table)

        self.summary = QLabel("No server running.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.enable_metrics_btn = QPushButton("Enable --metrics & relaunch")
        self.enable_metrics_btn.setVisible(False)
        self.enable_metrics_btn.clicked.connect(self.enable_metrics_requested)
        layout.addWidget(self.enable_metrics_btn)

        layout.addWidget(QLabel("Benchmark"))
        bench_config = QHBoxLayout()
        self.bench_sizes = QLineEdit("128, 512, 2048")
        bench_config.addWidget(self.bench_sizes)
        self.bench_npredict = QSpinBox()
        self.bench_npredict.setRange(1, 1_000_000)
        self.bench_npredict.setValue(128)
        bench_config.addWidget(self.bench_npredict)
        self.bench_warmup = QSpinBox()
        self.bench_warmup.setRange(0, 100)
        self.bench_warmup.setValue(1)
        bench_config.addWidget(self.bench_warmup)
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
        layout.addWidget(self.bench_table)
        self.bench_history = QListWidget()
        self.bench_history.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.bench_history)

        layout.addWidget(QLabel("Logs:"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        layout.addWidget(self.log_view, 1)
        self._last = ""
        self._draft = None
        self._log_buf = ""
        self.mtp_label = QLabel("")
        self.mtp_label.setVisible(False)
        layout.insertWidget(1, self.mtp_label)   # right after the summary label
        self._tok_history = deque(maxlen=60)
        self.throughput_label = QLabel("")
        self.throughput_label.setVisible(False)
        layout.insertWidget(1, self.throughput_label)
        self.endpoints_label = QLabel("")
        self.endpoints_label.setWordWrap(True)
        self.endpoints_label.setVisible(False)
        layout.insertWidget(1, self.endpoints_label)
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setVisible(False)
        layout.insertWidget(1, self.info_label)
        self._bench_running = False

    def _summary_text(self) -> str:
        return self._last

    def set_endpoints(self, port: int, embeddings: bool, reranking: bool):
        base = f"http://127.0.0.1:{port}"
        urls = []
        if embeddings:
            urls.append(f"embeddings: {base}/v1/embeddings")
        if reranking:
            urls.append(f"rerank: {base}/v1/rerank")
        if urls:
            self.endpoints_label.setText("Endpoints —  " + "    ".join(urls))
            self.endpoints_label.setVisible(True)
        else:
            self.endpoints_label.setText("")
            self.endpoints_label.setVisible(False)

    def set_props(self, info) -> None:
        """Render read-only /props info; hide if info is None or empty."""
        if info is None:
            self.info_label.setVisible(False)
            return
        parts = []
        if info.build:
            parts.append(f"build {info.build}")
        if info.n_ctx is not None:
            parts.append(f"ctx {info.n_ctx}")
        if info.modalities:
            mods = " ".join(f"{k}{'✓' if v else '✗'}"
                            for k, v in info.modalities.items())
            parts.append(mods)
        if info.model_alias:
            parts.append(f"alias {info.model_alias}")
        if info.total_slots is not None:
            parts.append(f"{info.total_slots} slots")
        if not parts:
            self.info_label.setVisible(False)
            return
        self.info_label.setText("Info —  " + " · ".join(parts))
        self.info_label.setVisible(True)

    def update_stats(self, data: dict):
        metrics_on = bool(data.get("metrics_on"))
        self.enable_metrics_btn.setVisible(not metrics_on)
        if not metrics_on:
            speed = "throughput: (enable --metrics to see tok/s)"
        else:
            tok = data.get("tok_s")
            ptok = data.get("prompt_tok_s")
            speed = f"gen {tok:.1f} tok/s" if tok is not None else "gen –"
            if ptok is not None:
                speed += f"  ·  prompt {ptok:.0f} tok/s"
        kv = data.get("kv_pct")
        kv_s = f"KV {kv * 100:.0f}%" if kv is not None else "KV –"
        parts = [speed, kv_s]
        if data.get("speculating"):
            parts.append("spec ●")
        for g in data.get("gpus", []):
            parts.append(f"{g.name}: {g.mem_used_mib}/{g.mem_total_mib} MiB, GPU {g.util_pct}%, {g.temp_c}°C")
        if data.get("cpu") or data.get("mem"):
            parts.append(f"container CPU {data.get('cpu','')} · MEM {data.get('mem','')}")
        if data.get("uptime"):
            parts.append(f"uptime {data['uptime']}")
        self._last = "    ".join(parts)
        self.summary.setText(self._last)
        tok = data.get("tok_s")
        if metrics_on and tok is not None:
            self._tok_history.append(tok)
            self.throughput_label.setText(f"gen tok/s  {sparkline(self._tok_history)}  {tok:.0f}")
            self.throughput_label.setVisible(True)
        else:
            self.throughput_label.setVisible(False)

    def append_log(self, text: str):
        self.log_view.appendPlainText(text.rstrip("\n"))
        self._log_buf += text
        *lines, self._log_buf = self._log_buf.split("\n")
        for line in lines:
            stats = parse_draft_stats(line)
            if stats is not None:
                self._draft = stats
                self.mtp_label.setText(self._mtp_text(stats, "log"))
                self.mtp_label.setVisible(True)

    def set_draft_stats(self, stats, source: str = "counters") -> None:
        """Show speculative-decode acceptance from a source other than the log.

        In router mode the log belongs to the router process, so a child's
        "draft acceptance = ..." line cannot be attributed to a model; the
        /metrics counters can, via ?model=<id>. The label says which source it
        used so the two are never confused.
        """
        if stats is None:
            return
        self._draft = stats
        self.mtp_label.setText(self._mtp_text(stats, source))
        self.mtp_label.setVisible(True)

    @staticmethod
    def _mtp_text(d, source: str = "log") -> str:
        pos = " / ".join(f"{p * 100:.0f}%" for p in d.per_position)
        return (f"MTP  accept {d.acceptance * 100:.0f}%  ·  len {d.mean_len:.2f}  "
                f"·  pos {pos}  ({source})")

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
        self._draft = None
        self._log_buf = ""
        self.mtp_label.setText("")
        self.mtp_label.setVisible(False)
        self.log_view.clear()
        self._tok_history.clear()
        self.throughput_label.setText("")
        self.throughput_label.setVisible(False)
        self.endpoints_label.setText("")
        self.endpoints_label.setVisible(False)
        self.info_label.setText("")
        self.info_label.setVisible(False)
        self.bench_table.setRowCount(0)
        self.bench_history.clear()
        self.bench_progress.setText("")

    def set_instances(self, rows, selected_name=None) -> None:
        from PySide6.QtWidgets import QTableWidgetItem, QPushButton
        self._instance_names = [r["name"] for r in rows]
        self.instances_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self.instances_table.setItem(i, 0, QTableWidgetItem(r["profile"]))
            self.instances_table.setItem(i, 1, QTableWidgetItem(str(r["port"] or "")))
            dot = "●" if r["health"] == "ready" else ("◐" if r["running"] else "○")
            self.instances_table.setItem(i, 2, QTableWidgetItem(f"{dot} {r['health']}"))
            self.instances_table.setItem(i, 3, QTableWidgetItem(r["stat"]))
            btn = QPushButton("■")
            btn.setEnabled(r["running"])
            btn.clicked.connect(lambda _=False, n=r["name"]: self.instance_stop_requested.emit(n))
            self.instances_table.setCellWidget(i, 4, btn)
        if selected_name in self._instance_names:
            self.instances_table.selectRow(self._instance_names.index(selected_name))

    def selected_instance_name(self):
        items = self.instances_table.selectionModel().selectedRows()
        if not items:
            return None
        return self._instance_names[items[0].row()]

    def _emit_selected_for_row(self, row) -> None:
        if 0 <= row < len(self._instance_names):
            self.instance_selected.emit(self._instance_names[row])
