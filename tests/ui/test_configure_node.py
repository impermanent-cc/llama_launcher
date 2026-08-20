from llama_launcher.core.nodes import Node
from llama_launcher.store.nodes import add_node
from llama_launcher.core.spec import Profile, Runtime


def test_node_combo_lists_local_plus_saved_nodes(main_window):
    add_node(Node(name="box-b", kind="remote", connection="box-b",
                  ssh_target="me@10.0.0.2"), main_window.base_dir())
    panel = main_window._configure_panel
    panel.reload_nodes()
    labels = [panel.node_combo.itemData(i) for i in range(panel.node_combo.count())]
    assert "local" in labels and "box-b" in labels


def test_node_round_trips_through_form(main_window):
    add_node(Node(name="box-b", kind="remote", connection="box-b",
                  ssh_target="me@10.0.0.2"), main_window.base_dir())
    panel = main_window._configure_panel
    panel.reload_nodes()
    panel.load_profile(Profile(name="p", runtime=Runtime(node="box-b")))
    assert panel.current_profile().runtime.node == "box-b"


def test_selecting_remote_node_flips_bind_to_lan(main_window):
    # A remote server must publish on a LAN interface; loopback is unreachable
    # from the GUI host. Picking a remote node flips 127.0.0.1 -> 0.0.0.0.
    add_node(Node(name="box-b", kind="remote", connection="box-b",
                  ssh_target="me@10.0.0.2"), main_window.base_dir())
    panel = main_window._configure_panel
    panel.reload_nodes()
    panel.node_combo.setCurrentIndex(panel.node_combo.findData("box-b"))
    assert panel.bind_host_combo.currentText() == "0.0.0.0"
