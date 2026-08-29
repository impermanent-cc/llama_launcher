import json
import shutil
import subprocess
from dataclasses import dataclass


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
    """Command head, with the remote-connection flag when set. podman selects a
    remote host with `--connection`, docker with `--context` -- the same concept,
    different flag, so a docker node was previously unreachable."""
    if not connection:
        return [binary]
    flag = "--context" if binary == "docker" else "--connection"
    return [binary, flag, connection]


def binary_available(binary: str) -> bool:
    return shutil.which(binary) is not None


def is_rootless(binary: str) -> bool:
    # podman exposes a boolean at Host.Security.Rootless; docker info has no such
    # field (the template errors, leaving stdout empty -> always False, so a
    # rootless docker daemon was misreported as rootful in the diagnostic
    # report). docker instead lists "name=rootless" among SecurityOptions -- use
    # a runtime-appropriate probe, like _base/node_reachable do.
    if binary == "docker":
        res = _run([binary, "info", "--format",
                    "{{range .SecurityOptions}}{{println .}}{{end}}"])
        return "rootless" in res.stdout
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
    # `container exists` is a podman-only convenience subcommand (docker has no
    # equivalent -> always False, breaking stale-container cleanup on docker).
    # `inspect --type container` returns 0/1 identically on both runtimes.
    return _run([*_base(binary, connection), "inspect", "--type", "container",
                 name]).returncode == 0


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
    # podman and docker name the same fields differently: podman uses
    # cpu_percent/mem_usage, docker uses CPUPerc/MemUsage. Reading only the
    # docker names left live CPU/MEM blank on podman -- the DEFAULT runtime.
    return {"cpu_perc": row.get("cpu_percent") or row.get("CPUPerc", ""),
            "mem_usage": row.get("mem_usage") or row.get("MemUsage", "")}


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


@dataclass
class ImageInfo:
    """Image metadata: tag, size, and creation time."""
    tag: str
    size: str
    created: str


def parse_images_detailed(output: str) -> dict[str, ImageInfo]:
    """Parse `<binary> images --format {{.Repository}}:{{.Tag}}|{{.Size}}|{{.CreatedAt}}`
    lines into a dict mapping full image tag to ImageInfo. Skips malformed lines
    and <none> repositories, {} on error."""
    result: dict[str, ImageInfo] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != 3:
            continue
        tag, size, created = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if "<none>" in tag:
            continue
        result[tag] = ImageInfo(tag=tag, size=size, created=created)
    return result


def list_images_detailed(binary: str, connection: str = "") -> dict[str, ImageInfo]:
    """Locally-pulled images with metadata (`<binary> images`), {} on error."""
    res = _run([*_base(binary, connection), "images", "--format",
                "{{.Repository}}:{{.Tag}}|{{.Size}}|{{.CreatedAt}}"])
    if res.returncode != 0:
        return {}
    return parse_images_detailed(res.stdout)


def remove_image(binary: str, tag: str, connection: str = "") -> tuple[bool, str]:
    """Remove an image by tag (`<binary> rmi <tag>`), returns (success, stderr)."""
    res = _run([*_base(binary, connection), "rmi", tag])
    return (res.returncode == 0, res.stderr)


_PROFILE_LABEL = "llama-launcher.profile"
_MODE_LABEL = "llama-launcher.mode"


def _parse_ps_payload(output: str) -> list[dict]:
    """Rows from `ps --format json` for either runtime: podman's single JSON
    array, or docker's newline-delimited bare objects (NDJSON)."""
    text = (output or "").strip()
    if not text:
        return []
    # podman: one array. (A docker single-container payload with no newline is
    # also valid JSON, but a dict, handled by the line loop below.)
    if text.startswith("["):
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return []
        return data if isinstance(data, list) else []
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _normalize_labels(labels) -> dict:
    """Labels as a dict. podman gives a dict; docker gives a comma-separated
    `k=v,k=v` string (or "")."""
    if isinstance(labels, dict):
        return labels
    if isinstance(labels, str) and labels:
        out: dict = {}
        for part in labels.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                out[k.strip()] = v.strip()
        return out
    return {}


def parse_ps_json(output: str) -> list[dict]:
    """Normalise `podman ps -a --format json` rows this launcher owns.

    Containers created before labels existed are still adopted, by falling back
    to the `llama-` name prefix. Anything else is ignored.

    Accepts both podman's single JSON array and docker's NDJSON (one bare object
    per line): reading only the array shape left the whole Monitor/Instances
    panel empty on docker even though the launch had succeeded.
    """
    data = _parse_ps_payload(output)

    rows: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        names = item.get("Names")
        name = names[0] if isinstance(names, list) and names else names
        if not isinstance(name, str) or not name:
            continue

        labels = _normalize_labels(item.get("Labels"))
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
    # `image exists` is podman-only; `image inspect` returns 0/1 on both runtimes.
    return _run([*_base(binary, connection), "image", "inspect", image]).returncode == 0


def pull_argv(image: str, binary: str, connection: str = "") -> list[str]:
    return [*_base(binary, connection), "pull", image]


def connection_add_argv(name: str, ssh_target: str, binary: str = "podman") -> list[str]:
    # podman registers a remote host as a named connection; docker registers it
    # as a named context over the same ssh transport.
    if binary == "docker":
        return [binary, "context", "create", name, "--docker", f"host=ssh://{ssh_target}"]
    return [binary, "system", "connection", "add", name, f"ssh://{ssh_target}"]


def connection_remove_argv(name: str, binary: str = "podman") -> list[str]:
    if binary == "docker":
        return [binary, "context", "rm", name]
    return [binary, "system", "connection", "remove", name]


def node_reachable(connection: str, binary: str = "podman") -> bool:
    # {{.Host.Arch}} is a podman info field; docker info has no such path, so use
    # a runtime-appropriate probe (both return 0 only when the daemon answers).
    fmt = "{{.OSType}}" if binary == "docker" else "{{.Host.Arch}}"
    return _run([*_base(binary, connection), "info", "--format", fmt]).returncode == 0
