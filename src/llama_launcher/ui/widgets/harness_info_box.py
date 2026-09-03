from PySide6.QtWidgets import QLabel, QPlainTextEdit, QVBoxLayout, QWidget


class HarnessInfoBox(QWidget):
    """Read-only 'server info to copy' block. ~3-4 lines tall."""

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(QLabel("Harness setup"))
        self.harness_text = QPlainTextEdit()
        self.harness_text.setReadOnly(True)
        self.harness_text.setMaximumHeight(90)  # ~3-4 lines
        root.addWidget(self.harness_text)

    def set_endpoint(self, base_url: str, model_ids: list) -> None:
        ids = "\n".join(f"  - {m}" for m in model_ids) or "  (no members yet)"
        self.harness_text.setPlainText(
            f"base_url: {base_url}/v1\n"
            f"api_key:  (see above, reveal to copy)\n"
            f"model ids:\n{ids}\n"
            f"\nCold loads can take minutes: raise your harness's request and "
            f"stale timeouts accordingly."
        )
