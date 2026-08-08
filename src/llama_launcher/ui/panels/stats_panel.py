from collections import deque

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from llama_launcher.core.mtp_stats import sparkline


def _gib_from_bytes(b: int) -> float:
    return b / 1024 ** 3


def _gib_from_mib(mib: int) -> float:
    return mib / 1024


class StatsPanel(QWidget):
    """Renders a StatsSnapshot: GPU (nvtop-like), System (btop-like), Container.

    Fed one snapshot at a time via update_stats(); keeps small rolling histories
    for the sparklines. Pure display -- no polling here (see StatsWorker).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.gpu_label = QLabel("GPU —")
        self.system_label = QLabel("System —")
        self.container_label = QLabel("Container —")
        for w in (self.gpu_label, self.system_label, self.container_label):
            w.setWordWrap(True)
            w.setTextInteractionFlags(w.textInteractionFlags())
            layout.addWidget(w)
        layout.addStretch(1)
        self._gpu_util_hist = {}          # gpu index -> deque[int]
        self._cpu_hist = deque(maxlen=40)

    def update_stats(self, snap) -> None:
        self._render_gpu(snap)
        self._render_system(snap)
        self._render_container(snap)

    def _render_gpu(self, snap) -> None:
        if not snap.gpu_available:
            self.gpu_label.setText("GPU — unavailable (nvidia-smi not found)")
            return
        blocks = []
        for i, g in enumerate(snap.gpus):
            hist = self._gpu_util_hist.setdefault(i, deque(maxlen=40))
            hist.append(g.util_pct)
            spark = sparkline(list(hist))
            power = f"  {g.power_draw_w:.0f}W" if g.power_draw_w is not None else ""
            blocks.append(
                f"{g.name}\n"
                f"  util {spark} {g.util_pct}%\n"
                f"  vram {_gib_from_mib(g.mem_used_mib):.1f} / "
                f"{_gib_from_mib(g.mem_total_mib):.1f} GiB  {g.temp_c}°C{power}")
        self.gpu_label.setText("GPU\n" + "\n".join(blocks))

    def _render_system(self, snap) -> None:
        if snap.cpu is None or snap.mem is None:
            self.system_label.setText("System — unavailable")
            return
        self._cpu_hist.append(snap.cpu.overall_pct)
        spark = sparkline(list(self._cpu_hist))
        cores = sparkline(snap.cpu.per_core_pct) if snap.cpu.per_core_pct else ""
        l1, l5, l15 = snap.cpu.load
        self.system_label.setText(
            f"System\n"
            f"  cpu  {spark} {snap.cpu.overall_pct:.0f}%\n"
            f"  cores {cores}\n"
            f"  ram  {_gib_from_bytes(snap.mem.used_bytes):.1f} / "
            f"{_gib_from_bytes(snap.mem.total_bytes):.1f} GiB\n"
            f"  load {l1:.2f} {l5:.2f} {l15:.2f}")

    def _render_container(self, snap) -> None:
        c = snap.container
        if c is None:
            self.container_label.setText("Container — no server running")
            return
        limit = (f" / {_gib_from_bytes(c.mem_limit_bytes):.1f} GiB"
                 if c.mem_limit_bytes else "")
        self.container_label.setText(
            f"Container: {c.name}\n"
            f"  cpu  {c.cpu_pct:.0f}%\n"
            f"  mem  {_gib_from_bytes(c.mem_used_bytes):.1f} GiB{limit}")
