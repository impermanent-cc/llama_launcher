"""A search field that jumps to a Configure setting or group by name."""

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QWidget


@dataclass(frozen=True)
class SearchEntry:
    """One jump target: the texts a query is matched against, whether it is
    a group or a setting, the widget to scroll to and tint, and a
    callable saying whether it is visible under the current mode and
    engine."""

    texts: tuple
    kind: str
    widget: object
    is_visible: object


def _norm(text: str) -> str:
    return text.strip().lstrip("-").lower()


def match_positions(query: str, entries) -> list:
    """Indices of the visible entries one of whose texts contains the query
    as a case-insensitive substring, leading dashes on the query and the
    texts ignored, in entry order; nothing for a blank query."""
    q = _norm(query)
    if not q:
        return []
    return [
        i
        for i, e in enumerate(entries)
        if e.is_visible() and any(q in _norm(t) for t in e.texts)
    ]


class _SearchEdit(QLineEdit):
    """The field: Enter steps to the next match, Shift+Enter to the
    previous, Escape clears."""

    next_requested = Signal()
    previous_requested = Signal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                self.previous_requested.emit()
            else:
                self.next_requested.emit()
            return
        if event.key() == Qt.Key_Escape:
            self.clear()
            return
        super().keyPressEvent(event)


class SettingSearchBar(QWidget):
    """A search field with a match counter that jumps through the settings
    and groups of the Configure tab; the host does the scrolling and the
    tinting on `jumped` and clears them on `cleared`."""

    jumped = Signal(object)
    cleared = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries: list = []
        self._matches: list = []
        self._pos = -1
        self.edit = _SearchEdit()
        self.edit.setPlaceholderText("Search settings")
        self.edit.setClearButtonEnabled(True)
        self.counter = QLabel("")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.edit, 1)
        row.addWidget(self.counter)
        self.edit.textChanged.connect(self.refresh)
        self.edit.next_requested.connect(self.next_match)
        self.edit.previous_requested.connect(self.previous_match)

    def set_entries(self, entries) -> None:
        self._entries = list(entries)
        self.refresh()

    def current(self):
        if 0 <= self._pos < len(self._matches):
            return self._entries[self._matches[self._pos]]
        return None

    def refresh(self) -> None:
        """Recompute the matches for the current text, staying on the entry
        already selected when it still matches, else jumping to the first."""
        kept = self.current()
        self.cleared.emit()
        self._matches = match_positions(self.edit.text(), self._entries)
        self._pos = 0 if self._matches else -1
        if kept is not None:
            for i, idx in enumerate(self._matches):
                if self._entries[idx] is kept:
                    self._pos = i
                    break
        self._announce()

    def next_match(self) -> None:
        self._step(1)

    def previous_match(self) -> None:
        self._step(-1)

    def _step(self, delta: int) -> None:
        if not self._matches:
            return
        self.cleared.emit()
        self._pos = (self._pos + delta) % len(self._matches)
        self._announce()

    def _announce(self) -> None:
        if not _norm(self.edit.text()):
            self.counter.setText("")
            return
        if not self._matches:
            self.counter.setText("no match")
            return
        self.counter.setText(f"{self._pos + 1} of {len(self._matches)}")
        self.jumped.emit(self.current())
