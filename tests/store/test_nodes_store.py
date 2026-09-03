from pathlib import Path

from llama_launcher.core.nodes import Node
from llama_launcher.store.nodes import (
    add_node,
    get_node,
    load_nodes,
    remove_node,
)


def test_empty_dir_yields_only_local(tmp_path: Path):
    nodes = load_nodes(tmp_path)
    assert [n.name for n in nodes] == ["local"]
    assert nodes[0].kind == "local"


def test_add_list_remove_round_trip(tmp_path: Path):
    add_node(
        Node(
            name="box-b",
            kind="remote",
            connection="box-b",
            ssh_target="me@10.0.0.2",
            binary="podman",
        ),
        tmp_path,
    )
    names = [n.name for n in load_nodes(tmp_path)]
    assert names == ["local", "box-b"]  # local always first
    assert get_node(tmp_path, "box-b").ssh_target == "me@10.0.0.2"
    remove_node("box-b", tmp_path)
    assert [n.name for n in load_nodes(tmp_path)] == ["local"]


def test_corrupt_file_falls_back_to_local(tmp_path: Path):
    (tmp_path / "nodes.json").write_text("{not json")
    assert [n.name for n in load_nodes(tmp_path)] == ["local"]


# -- gpu_ssh_target: node name -> ssh target for GPU probes ------------------


def test_gpu_ssh_target_local_and_missing(tmp_path):
    from llama_launcher.store.nodes import gpu_ssh_target

    assert gpu_ssh_target(tmp_path, "local") == ""
    assert gpu_ssh_target(tmp_path, "gone") == ""
    assert gpu_ssh_target(tmp_path, "") == ""


def test_gpu_ssh_target_remote(tmp_path):
    from llama_launcher.core.nodes import Node
    from llama_launcher.store.nodes import add_node, gpu_ssh_target

    add_node(
        Node(name="box-b", kind="remote", connection="box-b", ssh_target="me@10.0.0.2"),
        tmp_path,
    )
    assert gpu_ssh_target(tmp_path, "box-b") == "me@10.0.0.2"


# -- load-time trust clamps ---------------------------------------------------


def test_remotes_drops_rows_with_invalid_ssh_target(tmp_path):
    import json

    from llama_launcher.store.nodes import load_nodes

    (tmp_path / "nodes.json").write_text(
        json.dumps(
            [
                {"name": "ok", "ssh_target": "me@10.0.0.2", "binary": "podman"},
                {
                    "name": "evil",
                    "ssh_target": "-oProxyCommand=calc",
                    "binary": "podman",
                },
                {"name": "bad", "ssh_target": "we ird ; rm", "binary": "podman"},
            ]
        )
    )
    names = [n.name for n in load_nodes(tmp_path) if n.kind == "remote"]
    assert names == ["ok"]


def test_remotes_clamps_unknown_binary(tmp_path):
    import json

    from llama_launcher.store.nodes import load_nodes

    (tmp_path / "nodes.json").write_text(
        json.dumps(
            [
                {"name": "n", "ssh_target": "me@10.0.0.2", "binary": "/usr/bin/evil"},
            ]
        )
    )
    n = next(x for x in load_nodes(tmp_path) if x.kind == "remote")
    assert n.binary == "podman"


def test_load_drops_node_with_leading_dash_name(tmp_path):
    # A name/connection beginning with '-' could be misread by podman/docker as
    # an option flag (argv-flag confusion), so such a row is dropped on load.
    import json

    from llama_launcher.store.nodes import _nodes_file

    _nodes_file(tmp_path).write_text(
        json.dumps(
            [
                {
                    "name": "-oProxyCommand=x",
                    "connection": "-oProxyCommand=x",
                    "ssh_target": "me@10.0.0.9",
                    "binary": "podman",
                },
                {"name": "good", "connection": "good", "ssh_target": "me@10.0.0.8"},
            ]
        )
    )
    names = [n.name for n in load_nodes(tmp_path)]
    assert names == ["local", "good"]


def test_load_clamps_leading_dash_connection_to_name(tmp_path):
    import json

    from llama_launcher.store.nodes import _nodes_file

    _nodes_file(tmp_path).write_text(
        json.dumps(
            [
                {"name": "box", "connection": "-x", "ssh_target": "me@10.0.0.7"},
            ]
        )
    )
    box = get_node(tmp_path, "box")
    assert box is not None and box.connection == "box"  # not "-x"


def test_saved_nodes_file_is_owner_only(tmp_path):
    import stat

    from llama_launcher.store.nodes import _nodes_file

    add_node(
        Node(name="b", kind="remote", connection="b", ssh_target="me@1.2.3.4"), tmp_path
    )
    mode = stat.S_IMODE(_nodes_file(tmp_path).stat().st_mode)
    assert mode == 0o600
