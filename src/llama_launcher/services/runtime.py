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


def stop_argv(name: str, binary: str, timeout: int = 10) -> list[str]:
    """Argv to stop a container with an explicit grace period (for async/QProcess
    use so the UI thread never blocks on podman's stop timeout)."""
    return [binary, "stop", "-t", str(timeout), name]


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


_PROFILE_LABEL = "llama-launcher.profile"
_MODE_LABEL = "llama-launcher.mode"


def parse_ps_json(output: str) -> list[dict]:
    """Normalise `podman ps -a --format json` rows this launcher owns.

    Containers created before labels existed are still adopted, by falling back
    to the `llama-` name prefix. Anything else is ignored.
    """
    try:
        data = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, list):
        return []

    rows: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        names = item.get("Names")
        name = names[0] if isinstance(names, list) and names else names
        if not isinstance(name, str) or not name:
            continue

        labels = item.get("Labels")
        labels = labels if isinstance(labels, dict) else {}
        profile = labels.get(_PROFILE_LABEL)
        mode = labels.get(_MODE_LABEL)

        if not profile:
            if not name.startswith("llama-"):
                continue
            profile = name[len("llama-"):]
            mode = mode or "server"

        rows.append({
            "name": name,
            "running": str(item.get("State", "")).lower() == "running",
            "profile": profile,
            "mode": mode or "server",
        })
    return rows


def list_launcher_containers(binary: str) -> list[dict]:
    """Every container this launcher created, running or not."""
    res = _run([binary, "ps", "-a", "--filter", f"label={_PROFILE_LABEL}",
                "--format", "json"])
    if res.returncode != 0:
        return []
    return parse_ps_json(res.stdout)


def rm_argv(name: str, binary: str) -> list[str]:
    """Argv to remove a container (for the pre-launch cleanup of a stopped router)."""
    return [binary, "rm", "-f", name]
