from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton

_DOT = {"ready": "●", "starting": "◐", "loading": "◐"}


class StatCard(QFrame):
    """A compact live status card for one running launcher instance.

    Shows the profile/port, a health dot, a headline stat (gen tok/s, or "ready"
    for an embedding/rerank server, or "router" for a router) and KV%. Clickable
    to focus (emits `selected`); its ■ button emits `stop_requested`. Reused across
    ticks -- the owning panel calls update_row() to refresh labels in place.
    """
    selected = Signal(str)
    stop_requested = Signal(str)

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self._name = name
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumWidth(180)
        self._selected = False
        self.set_selected(False)
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        self._title = QLabel()
        self._title.setStyleSheet("font-weight: bold;")
        self._stop_btn = QPushButton("■")
        self._stop_btn.setFixedWidth(28)
        self._stop_btn.setToolTip("Stop this instance")
        self._stop_btn.clicked.connect(lambda: self.stop_requested.emit(self._name))
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

    def headline_text(self) -> str:
        return self._headline.text()

    def kv_text(self) -> str:
        return self._kv.text()

    def is_selected(self) -> bool:
        return self._selected

    def update_row(self, row: dict) -> None:
        port = row.get("port")
        title = row.get("profile") or self._name
        self._title.setText(f"{title}  :{port}" if port else title)
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
        self._selected = on
        self.setStyleSheet(
            "QFrame { border: 2px solid palette(highlight); border-radius: 4px; }" if on
            else "QFrame { border: 1px solid palette(mid); border-radius: 4px; }")

    def mousePressEvent(self, ev):
        self.selected.emit(self._name)
        super().mousePressEvent(ev)
