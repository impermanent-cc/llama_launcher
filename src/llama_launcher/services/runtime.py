import json
import shutil
import subprocess


def _run(args: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=False)
    except OSError as exc:
        return subprocess.CompletedProcess(args, returncode=127, stdout="", stderr=str(exc))


def binary_available(binary: str) -> bool:
    return shutil.which(binary) is not None


def is_rootless(binary: str) -> bool:
    res = _run([binary, "info", "--format", "{{.Host.Security.Rootless}}"])
    return res.stdout.strip() == "true"


def container_state(name: str, binary: str) -> str:
    res = _run([binary, "inspect", "-f", "{{.State.Running}}", name])
    if res.returncode != 0:
        return "absent"
    return "running" if res.stdout.strip() == "true" else "stopped"


def stop(name: str, binary: str) -> None:
    _run([binary, "stop", name])


def logs_argv(name: str, binary: str) -> list[str]:
    return [binary, "logs", "-f", name]


def container_exists(name: str, binary: str) -> bool:
    return _run([binary, "container", "exists", name]).returncode == 0


def started_at(name: str, binary: str) -> str | None:
    """Return the container's StartedAt timestamp (ISO 8601) or None on failure."""
    res = _run([binary, "inspect", "-f", "{{.State.StartedAt}}", name])
    if res.returncode != 0:
        return None
    val = res.stdout.strip()
    return val if val else None


def stats(name: str, binary: str) -> dict | None:
    res = _run([binary, "stats", "--no-stream", "--format", "json", name])
    if res.returncode != 0 or not res.stdout.strip():
        return None
    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError:
        return None
    row = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else None)
    if not row:
        return None
    return {"cpu_perc": row.get("CPUPerc", ""), "mem_usage": row.get("MemUsage", "")}


def parse_images(output: str) -> list[str]:
    """Filter `repo:tag` lines to llama.cpp images (dropping dangling/<none>),
    preserving order and de-duplicating."""
    seen: set[str] = set()
    out: list[str] = []
    for line in output.splitlines():
        ref = line.strip()
        if not ref or "<none>" in ref or "llama.cpp" not in ref:
            continue
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


def list_local_images(binary: str) -> list[str]:
    """Locally-pulled llama.cpp images (`<binary> images`), [] on error/missing binary."""
    res = _run([binary, "images", "--format", "{{.Repository}}:{{.Tag}}"])
    if res.returncode != 0:
        return []
    return parse_images(res.stdout)
