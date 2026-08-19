"""A running launcher server, joined from a container row + its stored profile.

Ports/endpoints live in the profile, not the container labels, so build_instances
joins list_launcher_containers() rows with stored profiles by name. Pure module.
"""
from dataclasses import dataclass

from llama_launcher.core.spec import DEFAULT_STOP_TIMEOUT, Profile
from llama_launcher.core.validation import dial_host


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


def build_instances(containers: list[dict], profiles: list[Profile]) -> list[Instance]:
    by_name = {p.name: p for p in profiles}
    out: list[Instance] = []
    for c in containers:
        prof = by_name.get(c.get("profile"))
        if prof is not None:
            port = prof.settings.get("port", 8080)
            host = dial_host(prof.runtime.bind_host)
            emb = bool(prof.settings.get("embeddings"))
            rer = bool(prof.settings.get("reranking"))
            stop_to = prof.runtime.stop_timeout
        else:
            port, host, emb, rer, stop_to = None, "127.0.0.1", False, False, DEFAULT_STOP_TIMEOUT
        out.append(Instance(
            name=c["name"], profile=c.get("profile", ""), mode=c.get("mode", "server"),
            running=bool(c.get("running")), port=port, host=host,
            embeddings=emb, reranking=rer, stop_timeout=stop_to))
    out.sort(key=lambda i: (not i.running, i.name))   # running first, then by name
    return out
