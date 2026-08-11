from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QDialog, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QPlainTextEdit, QPushButton, QRadioButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

_MASK = "••••••••••••"


def _status_text(model) -> str:
    if model.failed:
        code = "" if model.exit_code is None else f" (exit {model.exit_code})"
        return f"failed{code}"
    if model.progress is not None and model.status in ("loading", "downloading"):
        return f"{model.status} {model.progress * 100:.0f}%"
    return model.status


class RouterPanel(QWidget):
    """Router control plane: model list, per-row load/unload, harness setup."""

    load_requested = Signal(str)
    unload_requested = Signal(str)
    key_scope_changed = Signal(str)     # "global" | "own"
    key_saved = Signal(str, str)        # scope, value

    def __init__(self, parent=None):
        super().__init__(parent)
        self._api_key = ""
        self._base_url = ""
        self._revealed = False
        self._load_buttons: dict = {}
        self._unload_buttons: dict = {}

        root = QVBoxLayout(self)

        self.status_label = QLabel("disconnected")
        root.addWidget(self.status_label)

        self.banner = QLabel("")
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet("QLabel { color: #b35c00; font-weight: bold; }")
        self.banner.setVisible(False)
        root.addWidget(self.banner)

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

        scope_row = QHBoxLayout()
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
        root.addLayout(scope_row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Model id", "Status", ""])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        root.addWidget(self.table, 1)

        root.addWidget(QLabel("Harness setup"))
        self.harness_text = QPlainTextEdit()
        self.harness_text.setReadOnly(True)
        self.harness_text.setMaximumHeight(120)
        root.addWidget(self.harness_text)

    # -- state ---------------------------------------------------------------

    def set_connected(self, connected: bool) -> None:
        self.status_label.setText("● connected" if connected else "● disconnected")

    def set_error(self, text: str) -> None:
        """Report a failed control-plane action next to the buttons that caused it."""
        self.status_label.setText(f"● {text}" if text else "● connected")

    def set_exposure_warning(self, text: str) -> None:
        self.banner.setText(text)
        self.banner.setVisible(bool(text))

    def reveal_key(self, revealed: bool) -> None:
        self._revealed = bool(revealed)
        self.key_label.setText(self._api_key if self._revealed and self._api_key else _MASK)

    def _copy_key(self) -> None:
        from PySide6.QtWidgets import QApplication
        if self._api_key:
            QApplication.clipboard().setText(self._api_key)

    def set_endpoint(self, base_url: str, api_key: str, model_ids: list) -> None:
        self._base_url = base_url
        self._api_key = api_key or ""
        self.reveal_key(self._revealed)
        ids = "\n".join(f"  - {m}" for m in model_ids) or "  (no members yet)"
        self.harness_text.setPlainText(
            f"base_url: {base_url}/v1\n"
            f"api_key:  (see above — reveal to copy)\n"
            f"model ids:\n{ids}\n"
            f"\nCold loads can take minutes: raise your harness's request and "
            f"stale timeouts accordingly."
        )

    def set_models(self, models: list) -> None:
        self._load_buttons.clear()
        self._unload_buttons.clear()
        self.table.setRowCount(len(models))
        for row, model in enumerate(models):
            self.table.setItem(row, 0, QTableWidgetItem(model.id))
            self.table.setItem(row, 1, QTableWidgetItem(_status_text(model)))

            holder = QWidget()
            box = QHBoxLayout(holder)
            box.setContentsMargins(0, 0, 0, 0)

            load_btn = QPushButton("Load")
            load_btn.clicked.connect(
                lambda _checked=False, mid=model.id: self.load_requested.emit(mid))
            unload_btn = QPushButton("Unload")
            unload_btn.clicked.connect(
                lambda _checked=False, mid=model.id: self.unload_requested.emit(mid))

            load_btn.setEnabled(model.status in ("unloaded", "sleeping"))
            unload_btn.setEnabled(model.status in ("loaded", "sleeping", "loading"))

            box.addWidget(load_btn)
            box.addWidget(unload_btn)
            self.table.setCellWidget(row, 2, holder)

            self._load_buttons[model.id] = load_btn
            self._unload_buttons[model.id] = unload_btn

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
        dlg = _ApiKeyEditDialog(self._current_scope(), self)
        if dlg.exec():
            self._save_key(dlg.value())


class _ApiKeyEditDialog(QDialog):
    """Paste or generate a key value; validates before it can close."""

    def __init__(self, scope: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set API key")
        self._value = ""
        lay = QVBoxLayout(self)
        where = "shared global" if scope == "global" else "per-profile"
        lay.addWidget(QLabel(f"Set the {where} API key:"))
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
