import json
from dataclasses import asdict
from pathlib import Path

from llama_launcher.core.nodes import Node, LOCAL_NODE, valid_ssh_target

_ALLOWED_BINARIES = ("podman", "docker")


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
        # A loaded nodes.json is untrusted config: an ssh_target that isn't a
        # safe destination (e.g. `-oProxyCommand=...`) would be smuggled to ssh
        # as an option, and binary is used as argv[0]. Drop bad ssh rows, clamp
        # the binary.
        ssh = d.get("ssh_target", "")
        if not valid_ssh_target(ssh):
            continue
        binary = d.get("binary", "podman")
        if binary not in _ALLOWED_BINARIES:
            binary = "podman"
        out.append(Node(
            name=d["name"], kind="remote",
            connection=d.get("connection", d["name"]),
            ssh_target=ssh,
            binary=binary,
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


def gpu_ssh_target(base_dir: Path, name: str) -> str:
    """SSH target for GPU/memory probes on node `name` ('' = local/missing)."""
    node = get_node(Path(base_dir), name or "local")
    return node.ssh_target if node and node.kind == "remote" else ""
