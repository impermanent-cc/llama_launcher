"""A running launcher server, joined from a container row + its stored profile.

Ports/endpoints live in the profile, not the container labels, so build_instances
joins list_launcher_containers() rows with stored profiles by name. Pure module.
"""
import re
from dataclasses import dataclass

from llama_launcher.core.spec import DEFAULT_STOP_TIMEOUT, Profile, profile_port
from llama_launcher.core.validation import dial_host

_RPC_WORKER_SUFFIX = re.compile(r"-rpc(\d+)$")


@dataclass(frozen=True)
class Instance:
    name: str
    profile: str
    mode: str
    running: bool
    port: int | None
    host: str
    embeddings: bool
    reranking: bool
    stop_timeout: int = DEFAULT_STOP_TIMEOUT
    binary: str = "podman"
    kind: str = "container"
    pid: int | None = None
    node: str = "local"
    device: str = ""    # rpc-worker only: resolved from the pool profile's rpc_workers by index


def _worker_device(name: str, mode: str, prof: Profile | None) -> str:
    """The rpc-worker's device (e.g. "CUDA0"), resolved from the pool profile's
    `rpc_workers` list by matching the container name's `-rpcN` suffix to
    index N. Workers are labeled with the SAME `llama-launcher.profile` as
    their pool's head (so the pool joins as one profile), so `prof` here is
    the head's stored profile. "" when unresolvable (no match, no profile, or
    the index is out of range)."""
    if mode != "rpc-worker" or prof is None:
        return ""
    m = _RPC_WORKER_SUFFIX.search(name)
    if not m:
        return ""
    idx = int(m.group(1))
    workers = prof.runtime.rpc_workers
    return workers[idx].device if 0 <= idx < len(workers) else ""


def worker_card_title(inst: Instance) -> str:
    """Display title for an rpc-worker StatCard: "rpc-worker \u00b7 <node>[ \u00b7 <device>]".

    A worker container shares its pool head's `llama-launcher.profile` label,
    so the profile name/port cannot tell its card from the head's; the title
    carries the worker's own identity instead.
    """
    title = f"rpc-worker \u00b7 {inst.node}"
    if inst.device:
        title += f" \u00b7 {inst.device}"
    return title


def build_instances(containers: list[dict], profiles: list[Profile],
                    binary: str = "podman", node: str = "local",
                    node_host: str = "") -> list[Instance]:
    """`binary` is the container binary these rows were listed with; it's the
    fallback for an unmatched container (profile deleted) whose own binary is
    unknown. A matched container is controlled with its profile's binary."""
    by_name = {p.name: p for p in profiles}
    out: list[Instance] = []
    for c in containers:
        prof = by_name.get(c.get("profile"))
        mode = c.get("mode", "server")
        if prof is not None:
            port = profile_port(prof)
            host = node_host if (node != "local" and node_host) else dial_host(prof.runtime.bind_host)
            emb = bool(prof.settings.get("embeddings"))
            rer = bool(prof.settings.get("reranking"))
            stop_to = prof.runtime.stop_timeout
            bin_ = prof.runtime.binary
        else:
            port, emb, rer = None, False, False
            host = node_host if (node != "local" and node_host) else "127.0.0.1"
            stop_to, bin_ = DEFAULT_STOP_TIMEOUT, binary
        out.append(Instance(
            name=c["name"], profile=c.get("profile", ""), mode=mode,
            running=bool(c.get("running")), port=port, host=host,
            embeddings=emb, reranking=rer, stop_timeout=stop_to, binary=bin_,
            kind=c.get("kind", "container"), pid=c.get("pid"), node=node,
            device=_worker_device(c["name"], mode, prof)))
    out.sort(key=lambda i: (not i.running, i.name))   # running first, then by name
    return out
