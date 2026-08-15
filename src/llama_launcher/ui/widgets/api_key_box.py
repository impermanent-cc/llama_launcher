from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QDialog, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QRadioButton, QVBoxLayout, QWidget,
)

_MASK = "••••••••••••"


class ApiKeyBox(QWidget):
    """Router API key: masked value, reveal/copy/edit, and global/own scope."""

    key_scope_changed = Signal(str)     # "global" | "own"
    key_saved = Signal(str, str)        # scope, value

    def __init__(self, parent=None):
        super().__init__(parent)
        self._api_key = ""
        self._revealed = False
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("API key:"))
        self.key_label = QLabel(_MASK)
        self.key_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        key_row.addWidget(self.key_label, 1)
        self.reveal_check = QCheckBox("Reveal")
        self.reveal_check.toggled.connect(self.reveal_key)
        key_row.addWidget(self.reveal_check)
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.clicked.connect(self._copy_key)
        key_row.addWidget(self.copy_btn)
        self.edit_btn = QPushButton("Edit…")
        self.edit_btn.clicked.connect(self._open_edit)
        key_row.addWidget(self.edit_btn)
        root.addLayout(key_row)

        # The Global/Own scope selector only applies to the ROUTER's reusable
        # key. In single-server mode this box instead edits the server's
        # --api-key, so the scope row is hidden (see set_scope_visible).
        self._scope_applies = True
        self.scope_widget = QWidget()
        scope_row = QHBoxLayout(self.scope_widget)
        scope_row.setContentsMargins(0, 0, 0, 0)
        scope_row.addWidget(QLabel("Scope:"))
        self.scope_global = QRadioButton("Global")
        self.scope_own = QRadioButton("Own key for this profile")
        self._scope_group = QButtonGroup(self)
        self._scope_group.addButton(self.scope_global)
        self._scope_group.addButton(self.scope_own)
        self.scope_global.setChecked(True)
        # Connect ONE radio's toggled -> exactly one emit per change.
        self.scope_global.toggled.connect(self._on_scope_toggled)
        scope_row.addWidget(self.scope_global)
        scope_row.addWidget(self.scope_own)
        scope_row.addStretch(1)
        root.addWidget(self.scope_widget)

    def set_key(self, api_key: str) -> None:
        self._api_key = api_key or ""
        self.reveal_key(self._revealed)

    def set_scope_visible(self, visible: bool) -> None:
        """Show/hide the router-only Global/Own scope selector."""
        self._scope_applies = bool(visible)
        self.scope_widget.setVisible(visible)

    def reveal_key(self, revealed: bool) -> None:
        self._revealed = bool(revealed)
        self.key_label.setText(self._api_key if self._revealed and self._api_key else _MASK)

    def _copy_key(self) -> None:
        from PySide6.QtWidgets import QApplication
        if self._api_key:
            QApplication.clipboard().setText(self._api_key)

    # -- key scope -------------------------------------------------------------

    def _current_scope(self) -> str:
        return "own" if self.scope_own.isChecked() else "global"

    def set_scope(self, mode: str) -> None:
        self.scope_global.blockSignals(True)
        self.scope_own.blockSignals(True)
        (self.scope_own if mode == "own" else self.scope_global).setChecked(True)
        self.scope_global.blockSignals(False)
        self.scope_own.blockSignals(False)

    def _on_scope_toggled(self, _checked: bool) -> None:
        self.key_scope_changed.emit(self._current_scope())

    def _save_key(self, value: str) -> bool:
        """Normalize + emit key_saved for the current scope. False if invalid."""
        from llama_launcher.services.api_key import normalize_key
        try:
            key = normalize_key(value)
        except ValueError:
            return False
        self.key_saved.emit(self._current_scope(), key)
        return True

    def _open_edit(self) -> None:
        scope = self._current_scope() if self._scope_applies else None
        dlg = _ApiKeyEditDialog(scope, self)
        if dlg.exec():
            self._save_key(dlg.value())


class _ApiKeyEditDialog(QDialog):
    """Paste or generate a key value; validates before it can close."""

    def __init__(self, scope: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set API key")
        self._value = ""
        lay = QVBoxLayout(self)
        where = ("shared global" if scope == "global"
                 else "per-profile" if scope == "own" else None)
        label = f"Set the {where} API key:" if where else "Set the API key:"
        lay.addWidget(QLabel(label))
        self.field = QLineEdit()
        lay.addWidget(self.field)
        self.warn = QLabel("")
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet("QLabel { color: #b35c00; }")
        lay.addWidget(self.warn)
        self.field.textChanged.connect(self._check_warn)

        btns = QHBoxLayout()
        gen = QPushButton("Generate")
        gen.clicked.connect(self._generate)
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self._save)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btns.addWidget(gen)
        btns.addStretch(1)
        btns.addWidget(self.save_btn)
        btns.addWidget(cancel)
        lay.addLayout(btns)

    def _generate(self) -> None:
        from llama_launcher.services.api_key import generate_key
        self.field.setText(generate_key())

    def _check_warn(self, text: str) -> None:
        t = text.strip()
        self.warn.setText(
            "" if not t or t.startswith("sk-")
            else "Clients expecting OpenAI-style keys may reject a non 'sk-' key.")

    def _save(self) -> None:
        from llama_launcher.services.api_key import normalize_key
        try:
            self._value = normalize_key(self.field.text())
        except ValueError:
            self.warn.setText("Enter a non-empty, single-line key.")
            return
        self.accept()

    def value(self) -> str:
        return self._value
