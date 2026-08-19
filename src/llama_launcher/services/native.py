"""Native (non-container) llama-server processes: a per-instance JSON registry
under base_dir/native/, liveness via /proc, and helpers the Monitor/launch paths
call. Row shape mirrors runtime.list_launcher_containers so build_instances can
merge native + container instances into one Instance list."""
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from llama_launcher.core.command_builder import build_command
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


@dataclass
class NativeResult:
    ok: bool
    name: str
    host: str
    port: int
    pid: int | None = None
    error: str | None = None


def native_log_path(base_dir, profile_name: str) -> Path:
    return registry_dir(base_dir) / f"{slugify(profile_name)}.log"


def launch_native(profile, base_dir, now_iso: str) -> NativeResult:
    name = native_name(profile.name)
    host = profile.runtime.bind_host
    port = profile.settings.get("port", 8080)
    argv = build_command(profile)
    d = registry_dir(base_dir)
    d.mkdir(parents=True, exist_ok=True)
    log = native_log_path(base_dir, profile.name)
    try:
        with open(log, "w") as logf:
            proc = subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT,
                                    start_new_session=True)
    except OSError as exc:
        return NativeResult(False, name, host, port, None, str(exc))
    write_entry(base_dir, {"pid": proc.pid, "profile": profile.name, "port": port,
                           "host": host, "started_at": now_iso,
                           "binary": profile.runtime.native_binary, "log": str(log)})
    return NativeResult(True, name, host, port, proc.pid, None)


def stop_native(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def remove_native(name: str, base_dir) -> None:
    slug = name[len("llama-"):] if name.startswith("llama-") else name
    for path in (registry_dir(base_dir) / f"{slug}.json",
                 registry_dir(base_dir) / f"{slug}.log"):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


_CLK_TCK = os.sysconf("SC_CLK_TCK")


def _read_jiffies(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat", "r") as fh:
            fields = fh.read().split()
        # utime, stime are fields 14, 15 (1-indexed); index 13, 14 here.
        return int(fields[13]) + int(fields[14])
    except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
        return None


def _read_rss_bytes(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status", "r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024  # kB -> bytes
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return None
    return None


def proc_stats(pid: int, interval: float = 0.1) -> dict | None:
    start = _read_jiffies(pid)
    if start is None:
        return None
    if interval > 0:
        time.sleep(interval)
    end = _read_jiffies(pid)
    rss = _read_rss_bytes(pid)
    if end is None or rss is None:
        return None
    cpu = 0.0
    if interval > 0:
        cpu = max(0.0, (end - start) / (interval * _CLK_TCK) * 100.0)
    return {"cpu_perc": f"{cpu:.0f}%", "mem_usage": f"{rss / (1024 * 1024):.0f} MiB"}


def logs_argv(logpath: str) -> list[str]:
    return ["tail", "-n", "200", "-f", logpath]
