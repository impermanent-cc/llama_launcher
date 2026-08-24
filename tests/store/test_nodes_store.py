from pathlib import Path
from llama_launcher.core.nodes import Node
from llama_launcher.store.nodes import (
    load_nodes, save_nodes, get_node, add_node, remove_node,
)


def test_empty_dir_yields_only_local(tmp_path: Path):
    nodes = load_nodes(tmp_path)
    assert [n.name for n in nodes] == ["local"]
    assert nodes[0].kind == "local"


def test_add_list_remove_round_trip(tmp_path: Path):
    add_node(Node(name="box-b", kind="remote", connection="box-b",
                  ssh_target="me@10.0.0.2", binary="podman"), tmp_path)
    names = [n.name for n in load_nodes(tmp_path)]
    assert names == ["local", "box-b"]           # local always first
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
    add_node(Node(name="box-b", kind="remote", connection="box-b",
                  ssh_target="me@10.0.0.2"), tmp_path)
    assert gpu_ssh_target(tmp_path, "box-b") == "me@10.0.0.2"


# -- load-time trust clamps ---------------------------------------------------

def test_remotes_drops_rows_with_invalid_ssh_target(tmp_path):
    import json
    from llama_launcher.store.nodes import load_nodes
    (tmp_path / "nodes.json").write_text(json.dumps([
        {"name": "ok", "ssh_target": "me@10.0.0.2", "binary": "podman"},
        {"name": "evil", "ssh_target": "-oProxyCommand=calc", "binary": "podman"},
        {"name": "bad", "ssh_target": "we ird ; rm", "binary": "podman"},
    ]))
    names = [n.name for n in load_nodes(tmp_path) if n.kind == "remote"]
    assert names == ["ok"]


def test_remotes_clamps_unknown_binary(tmp_path):
    import json
    from llama_launcher.store.nodes import load_nodes
    (tmp_path / "nodes.json").write_text(json.dumps([
        {"name": "n", "ssh_target": "me@10.0.0.2", "binary": "/usr/bin/evil"},
    ]))
    n = [x for x in load_nodes(tmp_path) if x.kind == "remote"][0]
    assert n.binary == "podman"
