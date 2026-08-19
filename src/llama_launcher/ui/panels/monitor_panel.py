from collections import deque

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QScrollArea,
)

from llama_launcher.core.mtp_stats import parse_draft_stats, sparkline
from llama_launcher.ui.widgets.info_button import InfoButton
from llama_launcher.ui.widgets.stat_card import StatCard


class MonitorPanel(QWidget):
    enable_metrics_requested = Signal()
    instance_selected = Signal(str)
    instance_stop_requested = Signal(str)
    instance_remove_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # Cards row: one StatCard per running instance, side-by-side in a
        # horizontal scroll area so an empty/short list doesn't dominate the tab.
        self._cards: dict[str, StatCard] = {}
        self._selected_name: str | None = None
        self._cards_row = QHBoxLayout()
        self._cards_row.setSpacing(8)
        self._cards_row.addStretch(1)                 # keep cards left-packed
        cards_holder = QWidget()
        cards_holder.setLayout(self._cards_row)
        self._cards_scroll = QScrollArea()
        self._cards_scroll.setWidgetResizable(True)
        self._cards_scroll.setWidget(cards_holder)
        self._cards_scroll.setMaximumHeight(170)
        layout.insertWidget(0, self._cards_scroll)

        self.summary = QLabel("No server running.")
        self.summary.setWordWrap(True)
        # Persistent one-line key for the throughput/KV figures, which otherwise
        # read as bare numbers. gen/prompt come from llama.cpp's per-request
        # gauges, so they legitimately show 0 on an idle server between requests.
        # Kept as a hover tooltip + on-demand InfoButton popover instead of an
        # always-visible label, to avoid cluttering the tab with reminder text.
        _legend = ("gen / prompt = generation / prefill tok/s of the last request "
                   "(0 when idle between requests)  ·  KV = KV-cache used")
        # Deliberately NOT a tooltip on `summary`: the summary spans the whole
        # bar, so a tooltip there fires on hover anywhere along it, duplicating
        # the info button. The legend lives only on the compact ⓘ button.
        summary_row = QHBoxLayout()
        summary_row.addWidget(self.summary, 1)
        summary_row.addWidget(InfoButton(_legend))
        layout.addLayout(summary_row)
        self.enable_metrics_btn = QPushButton("Enable --metrics & relaunch")
        self.enable_metrics_btn.setVisible(False)
        self.enable_metrics_btn.clicked.connect(self.enable_metrics_requested)
        layout.addWidget(self.enable_metrics_btn)

        layout.addWidget(QLabel("Logs:"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        # The log is the primary payload of this tab; a floor plus stretch=1
        # lets it own the height (the benchmark lives on its own tab now).
        self.log_view.setMinimumHeight(240)
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

    def set_instance_cards(self, data: dict) -> None:
        rows = data.get("rows", [])
        self._selected_name = data.get("selected_name")
        names = [r["name"] for r in rows]
        if list(self._cards.keys()) != names:      # membership changed -> rebuild
            for card in self._cards.values():
                card.setParent(None)
                card.deleteLater()
            self._cards.clear()
            for name in names:
                card = StatCard(name)
                card.selected.connect(self.instance_selected)
                card.stop_requested.connect(self.instance_stop_requested)
                card.remove_requested.connect(self.instance_remove_requested)
                self._cards[name] = card
                self._cards_row.insertWidget(self._cards_row.count() - 1, card)  # before the stretch
        for r in rows:                              # update in place every tick
            card = self._cards[r["name"]]
            card.update_row(r)
            card.set_selected(r["name"] == self._selected_name)

    def card_names(self) -> list[str]:
        return list(self._cards.keys())

    def card(self, name: str):
        return self._cards.get(name)

    def selected_card_name(self) -> str | None:
        return self._selected_name if self._selected_name in self._cards else None

    def add_below_log(self, widget) -> None:
        self.layout().addWidget(widget)

    def add_status_banner(self, banner) -> None:
        self.layout().insertWidget(0, banner)   # above the cards row
