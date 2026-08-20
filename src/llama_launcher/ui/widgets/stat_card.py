from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton

_DOT = {"ready": "●", "starting": "◐", "loading": "◐"}


class StatCard(QFrame):
    """A compact live status card for one running launcher instance.

    Shows the profile/port, a health dot, a headline stat (gen tok/s, or "ready"
    for an embedding/rerank server, or "router" for a router) and KV%. Clickable
    to focus (emits `selected`). Its action button is dual-mode, carried over
    from the old instances table: running -> ■ emits `stop_requested`;
    stopped -> ✕ emits `remove_requested` (podman rm a dead container). Reused
    across ticks -- the owning panel calls update_row() to refresh labels in place.
    """
    selected = Signal(str)
    stop_requested = Signal(str)
    remove_requested = Signal(str)

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self._name = name
        self._running = True
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumWidth(180)
        self._selected = None       # sentinel: forces the initial paint below
        self.set_selected(False)
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        self._title = QLabel()
        self._title.setStyleSheet("font-weight: bold;")
        self._stop_btn = QPushButton("■")
        self._stop_btn.setFixedWidth(28)
        self._stop_btn.setToolTip("Stop this instance")
        self._stop_btn.clicked.connect(self._on_action)
        top.addWidget(self._title, 1)
        top.addWidget(self._stop_btn)
        v.addLayout(top)
        self._health = QLabel()
        self._headline = QLabel()
        self._kv = QLabel()
        for w in (self._health, self._headline, self._kv):
            v.addWidget(w)

    def name(self) -> str:
        return self._name

    def stop_button(self) -> QPushButton:
        return self._stop_btn

    def _on_action(self) -> None:
        (self.stop_requested if self._running else self.remove_requested).emit(self._name)

    def headline_text(self) -> str:
        return self._headline.text()

    def title_text(self) -> str:
        return self._title.text()

    def kv_text(self) -> str:
        return self._kv.text()

    def is_selected(self) -> bool:
        return self._selected

    def update_row(self, row: dict) -> None:
        running = row.get("running", True)
        self._running = running
        self._stop_btn.setText("■" if running else "✕")
        self._stop_btn.setToolTip("Stop this instance" if running else "Remove this stopped container")
        port = row.get("port")
        title = row.get("profile") or self._name
        title = f"{title}  :{port}" if port else title
        node = row.get("node")
        if node and node != "local":
            title = f"{title} · {node}"
        self._title.setText(title)
        health = row.get("health", "down")
        self._health.setText(f"{_DOT.get(health, '○')} {health}")
        if row.get("mode") == "router":
            headline = "router"
        elif row.get("embeddings") or row.get("reranking"):
            headline = "ready" if health == "ready" else health
        else:
            tok = row.get("tok_s")
            headline = f"{tok:.0f} tok/s" if tok else ("ready" if health == "ready" else health)
        self._headline.setText(headline)
        kv = row.get("kv_pct")
        self._kv.setText(f"KV {kv * 100:.0f}%" if kv is not None else "")

    def set_selected(self, on: bool) -> None:
        if on == self._selected:      # per-tick restyle guard (set_instance_cards
            return                    # calls this every poll even when unchanged)
        self._selected = on
        # Target the class name, not QFrame -- QLabel is a QFrame subclass, so a
        # bare "QFrame { border: ... }" selector cascaded onto every child label.
        self.setStyleSheet(
            "StatCard { border: 2px solid palette(highlight); border-radius: 4px; }" if on
            else "StatCard { border: 1px solid palette(mid); border-radius: 4px; }")

    def mousePressEvent(self, ev):
        self.selected.emit(self._name)
        super().mousePressEvent(ev)
