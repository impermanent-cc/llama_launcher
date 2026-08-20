import subprocess
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QComboBox,
    QPushButton, QListWidget, QLabel, QMessageBox,
)

from llama_launcher.core.nodes import Node, host_of, connection_for, valid_ssh_target
from llama_launcher.services import runtime, gpu
from llama_launcher.store.nodes import load_nodes, add_node, remove_node, get_node


def _run_ok(argv: list[str]) -> bool:
    """Run a short command, True on exit 0. Separated so tests can patch it."""
    try:
        return subprocess.run(argv, capture_output=True, text=True,
                              timeout=15).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


class NodesDialog(QDialog):
    def __init__(self, base_dir: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nodes")
        self._base_dir = base_dir

        root = QVBoxLayout(self)
        self.list = QListWidget()
        root.addWidget(self.list)

        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.ssh_edit = QLineEdit(); self.ssh_edit.setPlaceholderText("user@host[:port]")
        self.binary_combo = QComboBox(); self.binary_combo.addItems(["podman", "docker"])
        form.addRow("Name", self.name_edit)
        form.addRow("SSH target", self.ssh_edit)
        form.addRow("Binary", self.binary_combo)
        root.addLayout(form)

        btns = QHBoxLayout()
        add_btn = QPushButton("Add"); add_btn.clicked.connect(self._on_add)
        test_btn = QPushButton("Test"); test_btn.clicked.connect(self._on_test)
        rm_btn = QPushButton("Remove"); rm_btn.clicked.connect(self._on_remove)
        close_btn = QPushButton("Close"); close_btn.clicked.connect(self.accept)
        for b in (add_btn, test_btn, rm_btn, close_btn):
            btns.addWidget(b)
        root.addLayout(btns)

        self.status = QLabel("")
        root.addWidget(self.status)
        self._refresh_list()

    def _refresh_list(self) -> None:
        self.list.clear()
        for n in load_nodes(self._base_dir):
            suffix = "" if n.kind == "local" else f"  ({n.ssh_target})"
            self.list.addItem(f"{n.name}{suffix}")

    def add_node_from_fields(self, name: str, ssh_target: str, binary: str) -> None:
        if not name or not ssh_target:
            return
        # "local" is the implicit local node (store/nodes.py always prepends
        # it); a remote node reusing that name would be persisted alongside
        # it and mask it in the Configure "Node" dropdown / lookups by name.
        if name == "local":
            QMessageBox.warning(self, "Nodes",
                                "\"local\" is reserved for the local node.")
            return
        # Reject before ever touching runtime.connection_add_argv (which
        # would hand ssh_target to ssh/podman as an argv element) -- a target
        # starting with '-' could otherwise be smuggled in as an option flag.
        if not valid_ssh_target(ssh_target):
            QMessageBox.warning(self, "Nodes",
                                f"Invalid SSH target {ssh_target!r}.")
            return
        # Register the podman connection, then persist the node.
        if not _run_ok(runtime.connection_add_argv(name, ssh_target, binary)):
            QMessageBox.warning(self, "Nodes",
                                f"Could not add podman connection for {name!r}.")
            return
        add_node(Node(name=name, kind="remote", connection=name,
                      ssh_target=ssh_target, binary=binary), self._base_dir)
        self._refresh_list()

    def test_node(self, connection: str, ssh_target: str) -> tuple[bool, bool]:
        reachable = runtime.node_reachable(connection)
        gpus = bool(gpu.query_gpus(ssh_target)) if reachable else False
        return reachable, gpus

    def _on_add(self) -> None:
        self.add_node_from_fields(self.name_edit.text().strip(),
                                  self.ssh_edit.text().strip(),
                                  self.binary_combo.currentText())

    def _on_test(self) -> None:
        name = self.name_edit.text().strip()
        node = get_node(self._base_dir, name)
        conn = connection_for(node) if node else name
        target = node.ssh_target if node else self.ssh_edit.text().strip()
        reachable, gpus = self.test_node(conn, target)
        self.status.setText(
            f"{'reachable' if reachable else 'UNREACHABLE'} · "
            f"{'GPUs visible' if gpus else 'no GPUs'}")

    def _on_remove(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        name = item.text().split("  (")[0]
        if name == "local":
            return
        _run_ok(runtime.connection_remove_argv(name))
        remove_node(name, self._base_dir)
        self._refresh_list()
