from llama_launcher.core.nodes import Node
from llama_launcher.store.nodes import add_node
from llama_launcher.core.spec import Profile, Runtime


def test_connection_resolves_from_profile_node(main_window, tmp_path):
    add_node(Node(name="box-b", kind="remote", connection="box-b",
                  ssh_target="me@10.0.0.2"), main_window.base_dir())
    ctrl = main_window._launch
    local_p = Profile(name="a", runtime=Runtime(node="local"))
    remote_p = Profile(name="b", runtime=Runtime(node="box-b"))
    assert ctrl._connection_for_profile(local_p) == ""
    assert ctrl._connection_for_profile(remote_p) == "box-b"


def test_missing_node_falls_back_to_local(main_window):
    ctrl = main_window._launch
    p = Profile(name="c", runtime=Runtime(node="gone"))
    assert ctrl._connection_for_profile(p) == ""
