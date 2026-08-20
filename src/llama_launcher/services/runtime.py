import json
import shutil
import subprocess


_DEFAULT_TIMEOUT = 10        # seconds; bounds a hung container binary


def _run(args: list[str], timeout: float = _DEFAULT_TIMEOUT) -> subprocess.CompletedProcess:
    """Run a container-binary command, capturing output.

    A timeout is always passed: these run on the UI thread from update_status,
    so an unbounded call would freeze the GUI indefinitely if podman/docker
    hangs. On timeout (or a missing binary) a failing CompletedProcess is
    returned -- every caller already treats returncode != 0 as "absent/none".
    """
    try:
        return subprocess.run(args, capture_output=True, text=True, check=False,
                             timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(args, returncode=124, stdout="", stderr=str(exc))
    except OSError as exc:
        return subprocess.CompletedProcess(args, returncode=127, stdout="", stderr=str(exc))


def _base(binary: str, connection: str = "") -> list[str]:
    """Command head, with the podman remote-connection flag when set."""
    return [binary, "--connection", connection] if connection else [binary]


def binary_available(binary: str) -> bool:
    return shutil.which(binary) is not None


def is_rootless(binary: str) -> bool:
    res = _run([binary, "info", "--format", "{{.Host.Security.Rootless}}"])
    return res.stdout.strip() == "true"


def container_state(name: str, binary: str, connection: str = "") -> str:
    res = _run([*_base(binary, connection), "inspect", "-f", "{{.State.Running}}", name])
    if res.returncode != 0:
        return "absent"
    return "running" if res.stdout.strip() == "true" else "stopped"


def stop(name: str, binary: str) -> None:
    _run([binary, "stop", name])


def stop_argv(name: str, binary: str, timeout: int = 10, connection: str = "") -> list[str]:
    """Argv to stop a container with an explicit grace period (for async/QProcess
    use so the UI thread never blocks on podman's stop timeout)."""
    return [*_base(binary, connection), "stop", "-t", str(timeout), name]


def logs_argv(name: str, binary: str, connection: str = "") -> list[str]:
    return [*_base(binary, connection), "logs", "-f", name]


def container_exists(name: str, binary: str, connection: str = "") -> bool:
    return _run([*_base(binary, connection), "container", "exists", name]).returncode == 0


def started_at(name: str, binary: str, connection: str = "") -> str | None:
    """Return the container's StartedAt timestamp (ISO 8601) or None on failure."""
    res = _run([*_base(binary, connection), "inspect", "-f", "{{.State.StartedAt}}", name])
    if res.returncode != 0:
        return None
    val = res.stdout.strip()
    return val if val else None


def stats(name: str, binary: str, connection: str = "") -> dict | None:
    res = _run([*_base(binary, connection), "stats", "--no-stream", "--format", "json", name])
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


_ENGINE_IMAGE_MATCH = {
    "llama.cpp": ("llama.cpp",),
    "ik_llama.cpp": ("ik-llama-cpp", "ik_llama"),
}


def parse_images(output: str, engine: str = "llama.cpp") -> list[str]:
    """Filter `repo:tag` lines to the given engine's images (dropping
    dangling/<none>), preserving order and de-duplicating."""
    matches = _ENGINE_IMAGE_MATCH.get(engine, _ENGINE_IMAGE_MATCH["llama.cpp"])
    seen: set[str] = set()
    out: list[str] = []
    for line in output.splitlines():
        ref = line.strip()
        if not ref or "<none>" in ref or not any(m in ref for m in matches):
            continue
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


def list_local_images(binary: str, engine: str = "llama.cpp", connection: str = "") -> list[str]:
    """Locally-pulled images for `engine` (`<binary> images`), [] on error."""
    res = _run([*_base(binary, connection), "images", "--format", "{{.Repository}}:{{.Tag}}"])
    if res.returncode != 0:
        return []
    return parse_images(res.stdout, engine)


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


def list_launcher_containers(binary: str, connection: str = "") -> list[dict]:
    """Every container this launcher created, running or not."""
    res = _run([*_base(binary, connection), "ps", "-a", "--filter",
                f"label={_PROFILE_LABEL}", "--format", "json"])
    if res.returncode != 0:
        return []
    return parse_ps_json(res.stdout)


def rm_argv(name: str, binary: str, connection: str = "") -> list[str]:
    """Argv to remove a container (for the pre-launch cleanup of a stopped router)."""
    return [*_base(binary, connection), "rm", "-f", name]


def image_exists(image: str, binary: str, connection: str = "") -> bool:
    return _run([*_base(binary, connection), "image", "exists", image]).returncode == 0


def pull_argv(image: str, binary: str, connection: str = "") -> list[str]:
    return [*_base(binary, connection), "pull", image]


def connection_add_argv(name: str, ssh_target: str, binary: str = "podman") -> list[str]:
    return [binary, "system", "connection", "add", name, f"ssh://{ssh_target}"]


def connection_remove_argv(name: str, binary: str = "podman") -> list[str]:
    return [binary, "system", "connection", "remove", name]


def node_reachable(connection: str, binary: str = "podman") -> bool:
    return _run([*_base(binary, connection), "info", "--format", "{{.Host.Arch}}"]).returncode == 0
