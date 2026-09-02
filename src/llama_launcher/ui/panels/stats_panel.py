from collections import deque

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from llama_launcher.core.mtp_stats import sparkline

_SPARK_W = 40          # fixed sparkline width -> the panel never widens over time


def _gib_from_bytes(b: int) -> float:
    return b / 1024 ** 3


def _gib_from_mib(mib: int) -> float:
    return mib / 1024


class StatsPanel(QWidget):
    """Renders a StatsSnapshot: GPU (nvtop-like), System (btop-like), Container.

    Fed one snapshot at a time via update_stats(); keeps small rolling histories
    for the sparklines. Pure display -- no polling here (see StatsWorker).

    Width is kept CONSTANT across updates on purpose: a monospace font plus
    fixed-width sparklines (pre-filled to _SPARK_W) and fixed-width numeric
    fields. A sparkline that grows from 1 to _SPARK_W chars as history fills
    would widen the labels and make the QDockWidget (and the whole window)
    grow tick after tick without ever shrinking back.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        mono = QFont("monospace")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.gpu_label = QLabel("GPU: …")
        self.system_label = QLabel("System: …")
        self.container_label = QLabel("Container: …")
        for w in (self.gpu_label, self.system_label, self.container_label):
            w.setFont(mono)
            w.setWordWrap(True)
            layout.addWidget(w)
        layout.addStretch(1)
        self._hist: dict = {}          # metric key -> deque[float] (len == _SPARK_W)

    def _spark(self, key, value: float) -> str:
        """Push value into the key's rolling history and render a fixed-width
        sparkline. The deque is pre-filled on first use so the string is always
        _SPARK_W chars wide -- constant width from the very first update."""
        hist = self._hist.get(key)
        if hist is None:
            hist = deque([value] * _SPARK_W, maxlen=_SPARK_W)
            self._hist[key] = hist
        else:
            hist.append(value)
        return sparkline(list(hist))

    def update_stats(self, snap) -> None:
        self._render_gpu(snap)
        self._render_system(snap)
        self._render_container(snap)

    def _render_gpu(self, snap) -> None:
        if not snap.gpu_available:
            self.gpu_label.setText("GPU: unavailable (nvidia-smi not found)")
            return
        blocks = []
        for i, g in enumerate(snap.gpus):
            spark = self._spark(("gpu", i), g.util_pct)
            power = f"  {g.power_draw_w:4.0f}W" if g.power_draw_w is not None else ""
            blocks.append(
                f"{g.name}\n"
                f"  util {spark} {g.util_pct:3d}%\n"
                f"  vram {_gib_from_mib(g.mem_used_mib):5.1f} / "
                f"{_gib_from_mib(g.mem_total_mib):5.1f} GiB  {g.temp_c:3d}°C{power}")
        self.gpu_label.setText("GPU\n" + "\n".join(blocks))

    def _render_system(self, snap) -> None:
        if snap.cpu is None or snap.mem is None:
            self.system_label.setText("System: unavailable")
            return
        spark = self._spark("cpu", snap.cpu.overall_pct)
        cores = sparkline(snap.cpu.per_core_pct) if snap.cpu.per_core_pct else ""
        l1, l5, l15 = snap.cpu.load
        self.system_label.setText(
            f"System\n"
            f"  cpu   {spark} {snap.cpu.overall_pct:5.1f}%\n"
            f"  cores {cores}\n"
            f"  ram   {_gib_from_bytes(snap.mem.used_bytes):6.1f} / "
            f"{_gib_from_bytes(snap.mem.total_bytes):6.1f} GiB\n"
            f"  load  {l1:.2f} {l5:.2f} {l15:.2f}")

    def _render_container(self, snap) -> None:
        c = snap.container
        if c is None:
            self.container_label.setText("Container: no server running")
            return
        limit = (f" / {_gib_from_bytes(c.mem_limit_bytes):6.1f} GiB"
                 if c.mem_limit_bytes else "")
        self.container_label.setText(
            f"Container: {c.name}\n"
            f"  cpu   {c.cpu_pct:6.1f}%\n"
            f"  mem   {_gib_from_bytes(c.mem_used_bytes):6.1f} GiB{limit}")
