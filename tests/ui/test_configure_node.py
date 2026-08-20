from llama_launcher.core.nodes import Node
from llama_launcher.store.nodes import add_node
from llama_launcher.core.spec import Profile, Runtime, Mount
from llama_launcher.services import runtime as runtime_svc


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


def test_router_issues_warns_when_image_missing_on_remote_node(main_window, monkeypatch):
    """A profile pinned to a remote node whose image isn't present there should
    surface the image-present warning from validate(); a hermetic
    runtime.image_exists stub stands in for a real podman/ssh round trip."""
    add_node(Node(name="box-b", kind="remote", connection="box-b",
                  ssh_target="me@10.0.0.2"), main_window.base_dir())
    monkeypatch.setattr(runtime_svc, "image_exists",
                        lambda image, binary, connection="": False)
    panel = main_window._configure_panel
    panel.reload_nodes()
    panel.load_profile(Profile(
        name="p", image="img:tag", runtime=Runtime(node="box-b"),
        mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
        model="/models/m.gguf", settings={"port": 8080}))
    issues = panel.router_issues()
    assert any("not present on the selected node" in i.message for i in issues)


def test_router_issues_skips_image_check_for_local_node(main_window, monkeypatch):
    """A local profile never calls the remote image-present probe."""
    called = []
    monkeypatch.setattr(runtime_svc, "image_exists",
                        lambda image, binary, connection="": called.append(1) or True)
    panel = main_window._configure_panel
    panel.load_profile(Profile(
        name="p", image="img:tag", runtime=Runtime(node="local"),
        mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
        model="/models/m.gguf", settings={"port": 8080}))
    panel.router_issues()
    assert called == []
