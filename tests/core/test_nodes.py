from llama_launcher.core.nodes import Node, LOCAL_NODE, connection_for, host_of


def test_local_node_injects_no_connection():
    assert LOCAL_NODE.kind == "local"
    assert connection_for(LOCAL_NODE) == ""
    assert host_of(LOCAL_NODE) == "127.0.0.1"


def test_remote_node_connection_and_host():
    n = Node(name="box-b", kind="remote", connection="box-b",
             ssh_target="me@192.168.1.11:22", binary="podman")
    assert connection_for(n) == "box-b"
    assert host_of(n) == "192.168.1.11"
