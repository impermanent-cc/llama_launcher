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
