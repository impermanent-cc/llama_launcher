from llama_launcher.core.nodes import Node
from llama_launcher.store.nodes import add_node
from llama_launcher.ui.controllers.monitor_controller import focused_gpu_ssh


def test_local_node_has_no_ssh(tmp_path):
    assert focused_gpu_ssh("local", tmp_path) == ""


def test_remote_node_returns_ssh_target(tmp_path):
    add_node(Node(name="box-b", kind="remote", connection="box-b",
                  ssh_target="me@10.0.0.2"), tmp_path)
    assert focused_gpu_ssh("box-b", tmp_path) == "me@10.0.0.2"


def test_missing_node_has_no_ssh(tmp_path):
    assert focused_gpu_ssh("gone", tmp_path) == ""
