"""Registered machines the launcher can drive. Pure module (no I/O)."""

import re
from dataclasses import dataclass

# host is a name/IPv4 charset OR a bracketed IPv6 literal ([2001:db8::1]).
_SSH_TARGET_RE = re.compile(
    r"^(?:[A-Za-z0-9._-]+@)?(?:[A-Za-z0-9._-]+|\[[0-9A-Fa-f:]+\])(?::[0-9]+)?$"
)

# Options every launcher ssh probe passes so a first contact can't hang the UI
# on an interactive prompt: fail instead of prompting (BatchMode), bound the
# connect, and trust-on-first-use a new host key rather than blocking on it.
SSH_OPTS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "StrictHostKeyChecking=accept-new",
]


@dataclass(frozen=True)
class Node:
    name: str
    kind: str = "local"  # "local" | "remote"
    connection: str = ""  # podman connection name (remote only)
    ssh_target: str = ""  # user@host[:port] (remote only)
    binary: str = "podman"  # "podman" | "docker"
    enabled: bool = True


LOCAL_NODE = Node(name="local", kind="local")


def valid_ssh_target(target: str) -> bool:
    """True if `target` is a safe ssh destination (`user@host[:port]` or `host`).
    Rejects empty, a leading '-', and anything outside the charset, so it can
    never be smuggled to ssh as an option flag (argv-flag injection)."""
    return (
        bool(target)
        and not target.startswith("-")
        and bool(_SSH_TARGET_RE.match(target))
    )


def connection_for(node: Node) -> str:
    """The `--connection` name for podman, or '' for the local host."""
    return node.connection if node.kind == "remote" else ""


def host_of(node: Node) -> str:
    """The address to dial this node's servers on. Local -> loopback; remote ->
    the host portion of ssh_target (strip user@ and :port)."""
    if node.kind != "remote":
        return "127.0.0.1"
    target = node.ssh_target
    host = target.rsplit("@", 1)[-1] if "@" in target else target
    if host.startswith("["):
        # Bracketed IPv6 (`[2001:db8::1]` / `[2001:db8::1]:22`): the address is
        # inside the brackets and an optional :port follows the ']', so a plain
        # rsplit(':') would split INSIDE the address. Return the bare literal;
        # url_host re-brackets it for a URL authority.
        end = host.find("]")
        if end != -1:
            return host[1:end] or "127.0.0.1"
    host = host.rsplit(":", 1)[0] if ":" in host else host
    return host or "127.0.0.1"
