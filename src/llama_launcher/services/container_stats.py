import json
import subprocess
from dataclasses import dataclass

_UNITS = {
    "b": 1,
    "kb": 10**3,
    "mb": 10**6,
    "gb": 10**9,
    "tb": 10**12,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}


@dataclass(frozen=True)
class ContainerStat:
    name: str
    cpu_pct: float
    mem_used_bytes: int
    mem_limit_bytes: int | None


def parse_size(token: str) -> int | None:
    token = str(token).strip()
    num = ""
    for ch in token:
        if ch.isdigit() or ch == ".":
            num += ch
        else:
            break
    if not num:
        return None
    unit = token[len(num) :].strip().lower()
    try:
        return int(float(num) * _UNITS.get(unit, 1))
    except ValueError:
        return None


def _pct(token) -> float:
    try:
        return float(str(token).strip().rstrip("%"))
    except (ValueError, AttributeError):
        return 0.0


def parse_container_stats(output: str) -> ContainerStat | None:
    """First row of `<binary> stats --no-stream --format json`.

    podman keys: name / cpu_percent / mem_usage ("used / limit").
    docker keys: Name / CPUPerc / MemUsage. Returns None when empty/unparseable.
    """
    try:
        data = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return None
    row = data[0]
    name = row.get("name") or row.get("Name") or ""
    cpu = _pct(row.get("cpu_percent") or row.get("CPUPerc") or "0")
    mem = str(row.get("mem_usage") or row.get("MemUsage") or "")
    used_str, _, limit_str = mem.partition("/")
    return ContainerStat(
        name=name,
        cpu_pct=cpu,
        mem_used_bytes=parse_size(used_str) or 0,
        mem_limit_bytes=parse_size(limit_str),
    )


def query_container_stats(name: str, binary: str) -> ContainerStat | None:
    if not name:
        return None
    try:
        res = subprocess.run(
            [binary, "stats", "--no-stream", "--format", "json", name],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    return parse_container_stats(res.stdout)
