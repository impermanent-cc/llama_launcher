from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPlainTextEdit, QPushButton

from llama_launcher.core.mtp_stats import parse_draft_stats


class MonitorPanel(QWidget):
    enable_metrics_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.summary = QLabel("No server running.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.enable_metrics_btn = QPushButton("Enable --metrics & relaunch")
        self.enable_metrics_btn.setVisible(False)
        self.enable_metrics_btn.clicked.connect(self.enable_metrics_requested)
        layout.addWidget(self.enable_metrics_btn)
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

    def _summary_text(self) -> str:
        return self._last

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
        for g in data.get("gpus", []):
            parts.append(f"{g.name}: {g.mem_used_mib}/{g.mem_total_mib} MiB, GPU {g.util_pct}%, {g.temp_c}°C")
        if data.get("cpu") or data.get("mem"):
            parts.append(f"container CPU {data.get('cpu','')} · MEM {data.get('mem','')}")
        if data.get("uptime"):
            parts.append(f"uptime {data['uptime']}")
        self._last = "    ".join(parts)
        self.summary.setText(self._last)

    def append_log(self, text: str):
        self.log_view.appendPlainText(text.rstrip("\n"))
        self._log_buf += text
        *lines, self._log_buf = self._log_buf.split("\n")
        for line in lines:
            stats = parse_draft_stats(line)
            if stats is not None:
                self._draft = stats
                self.mtp_label.setText(self._mtp_text(stats))
                self.mtp_label.setVisible(True)

    @staticmethod
    def _mtp_text(d) -> str:
        pos = " / ".join(f"{p * 100:.0f}%" for p in d.per_position)
        return f"MTP  accept {d.acceptance * 100:.0f}%  ·  len {d.mean_len:.2f}  ·  pos {pos}"

    def reset(self):
        self._draft = None
        self._log_buf = ""
        self.mtp_label.setText("")
        self.mtp_label.setVisible(False)
        self.log_view.clear()
