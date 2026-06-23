from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPlainTextEdit


class MonitorPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.summary = QLabel("No server running.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        layout.addWidget(QLabel("Logs:"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        layout.addWidget(self.log_view, 1)
        self._last = ""

    def _summary_text(self) -> str:
        return self._last

    def update_stats(self, data: dict):
        if not data.get("metrics_on"):
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
