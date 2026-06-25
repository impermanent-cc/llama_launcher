from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QCheckBox, QDialogButtonBox, QLabel
)

from llama_launcher.core.report import REPORT_SECTIONS

_LABELS = {
    "command": "Command + profile (API key redacted)",
    "validation": "Validation + status history",
    "runtime": "Runtime + GPU + host",
    "metrics": "Server metrics (tok/s, KV cache)",
    "logs": "Image build + recent logs",
}


class ReportDialog(QDialog):
    def __init__(self, initial: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate diagnostic report")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Include sections:"))
        self._checks = {}
        for key in REPORT_SECTIONS:
            cb = QCheckBox(_LABELS[key])
            cb.setChecked(bool(initial.get(key, True)))
            self._checks[key] = cb
            layout.addWidget(cb)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def selected_sections(self) -> dict:
        return {k: cb.isChecked() for k, cb in self._checks.items()}
