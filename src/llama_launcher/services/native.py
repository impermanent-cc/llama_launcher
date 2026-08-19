"""Native (non-container) llama-server processes: a per-instance JSON registry
under base_dir/native/, liveness via /proc, and helpers the Monitor/launch paths
call. Row shape mirrors runtime.list_launcher_containers so build_instances can
merge native + container instances into one Instance list."""
import json
import os
from pathlib import Path

from llama_launcher.core.spec import slugify


def registry_dir(base_dir: Path) -> Path:
    return Path(base_dir) / "native"


def native_name(profile_name: str) -> str:
    return f"llama-{slugify(profile_name)}"


def _entry_path(base_dir: Path, profile_name: str) -> Path:
    return registry_dir(base_dir) / f"{slugify(profile_name)}.json"


def write_entry(base_dir: Path, entry: dict) -> Path:
    d = registry_dir(base_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = _entry_path(base_dir, entry["profile"])
    path.write_text(json.dumps(entry))
    return path


def read_entries(base_dir: Path) -> list[dict]:
    d = registry_dir(base_dir)
    if not d.exists():
        return []
    out: list[dict] = []
    for f in sorted(d.glob("*.json")):
        try:
            out.append(json.loads(f.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def is_alive(pid: int, binary: str) -> bool:
    """True iff /proc/<pid> exists and its cmdline references `binary` -- the
    cmdline check guards against a reused pid after a reboot naming a different
    process."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            cmdline = fh.read().replace(b"\0", b" ").decode(errors="replace")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
    return binary in cmdline


def list_native_instances(base_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for e in read_entries(base_dir):
        pid = e.get("pid")
        if isinstance(pid, int) and is_alive(pid, e.get("binary", "")):
            rows.append({"name": native_name(e["profile"]), "running": True,
                         "profile": e["profile"], "mode": "server",
                         "pid": pid, "kind": "native"})
        else:
            # Prune a dead/stale registry file so the card list can clear.
            p = _entry_path(base_dir, e.get("profile", ""))
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
    return rows
