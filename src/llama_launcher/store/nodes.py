import json
from dataclasses import asdict
from pathlib import Path

from llama_launcher.core.nodes import Node, LOCAL_NODE


def _nodes_file(base_dir: Path) -> Path:
    return base_dir / "nodes.json"


def _remotes(base_dir: Path) -> list[Node]:
    path = _nodes_file(base_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []                      # corrupt file -> no remotes (local still returned)
    if not isinstance(data, list):
        return []
    out: list[Node] = []
    for d in data:
        if not isinstance(d, dict) or not d.get("name"):
            continue
        out.append(Node(
            name=d["name"], kind="remote",
            connection=d.get("connection", d["name"]),
            ssh_target=d.get("ssh_target", ""),
            binary=d.get("binary", "podman"),
            enabled=bool(d.get("enabled", True)),
        ))
    return out


def load_nodes(base_dir: Path) -> list[Node]:
    """The local node (always first) followed by saved remote nodes."""
    return [LOCAL_NODE, *_remotes(base_dir)]


def save_nodes(nodes: list[Node], base_dir: Path) -> None:
    """Persist only the remote nodes; the local node is implicit."""
    base_dir.mkdir(parents=True, exist_ok=True)
    remotes = [asdict(n) for n in nodes if n.kind == "remote"]
    for r in remotes:
        r.pop("kind", None)            # implied "remote" on load
    _nodes_file(base_dir).write_text(json.dumps(remotes, indent=2))


def get_node(base_dir: Path, name: str) -> Node | None:
    return next((n for n in load_nodes(base_dir) if n.name == name), None)


def add_node(node: Node, base_dir: Path) -> None:
    remotes = [n for n in _remotes(base_dir) if n.name != node.name]
    save_nodes([*remotes, node], base_dir)


def remove_node(name: str, base_dir: Path) -> None:
    save_nodes([n for n in _remotes(base_dir) if n.name != name], base_dir)
