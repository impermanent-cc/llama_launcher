from unittest.mock import patch
from llama_launcher.ui.dialogs.nodes_dialog import NodesDialog
from llama_launcher.store.nodes import load_nodes


def test_add_persists_remote_node(tmp_path, qtbot):
    dlg = NodesDialog(tmp_path)
    qtbot.addWidget(dlg)
    with patch("llama_launcher.ui.dialogs.nodes_dialog.runtime.connection_add_argv",
               return_value=["true"]), \
         patch("llama_launcher.ui.dialogs.nodes_dialog._run_ok", return_value=True):
        dlg.add_node_from_fields("box-b", "me@10.0.0.2", "podman")
    assert "box-b" in [n.name for n in load_nodes(tmp_path)]


def test_test_node_reports_reachable_and_gpu(tmp_path, qtbot):
    dlg = NodesDialog(tmp_path)
    qtbot.addWidget(dlg)
    with patch("llama_launcher.ui.dialogs.nodes_dialog.runtime.node_reachable",
               return_value=True), \
         patch("llama_launcher.ui.dialogs.nodes_dialog.gpu.query_gpus",
               return_value=[object()]):
        reachable, gpus = dlg.test_node("box-b", "me@10.0.0.2")
    assert reachable is True and gpus is True


def test_invalid_ssh_target_is_not_persisted(tmp_path, qtbot):
    # Argv-flag injection: a target starting with '-' could be smuggled to ssh
    # as an option. valid_ssh_target() rejects it; add_node_from_fields must
    # refuse before ever calling runtime.connection_add_argv.
    dlg = NodesDialog(tmp_path)
    qtbot.addWidget(dlg)
    with patch("llama_launcher.ui.dialogs.nodes_dialog.runtime.connection_add_argv",
               return_value=["true"]) as add_argv, \
         patch("llama_launcher.ui.dialogs.nodes_dialog._run_ok", return_value=True), \
         patch("llama_launcher.ui.dialogs.nodes_dialog.QMessageBox.warning") as warn:
        dlg.add_node_from_fields("box-b", "-oProxyCommand=x", "podman")
    add_argv.assert_not_called()
    warn.assert_called_once()
    assert "box-b" not in [n.name for n in load_nodes(tmp_path)]


def test_node_named_local_is_not_persisted(tmp_path, qtbot):
    # "local" is the implicit local node; a remote node reusing that name
    # would mask it, so add_node_from_fields must refuse.
    dlg = NodesDialog(tmp_path)
    qtbot.addWidget(dlg)
    with patch("llama_launcher.ui.dialogs.nodes_dialog.runtime.connection_add_argv",
               return_value=["true"]) as add_argv, \
         patch("llama_launcher.ui.dialogs.nodes_dialog._run_ok", return_value=True), \
         patch("llama_launcher.ui.dialogs.nodes_dialog.QMessageBox.warning") as warn:
        dlg.add_node_from_fields("local", "me@10.0.0.2", "podman")
    add_argv.assert_not_called()
    warn.assert_called_once()
    remote_names = [n.name for n in load_nodes(tmp_path) if n.kind == "remote"]
    assert "local" not in remote_names
