import shutil
import subprocess
from dataclasses import dataclass

from llama_launcher.core.nodes import valid_ssh_target, SSH_OPTS

_QUERY = ("memory.used,memory.total,memory.free,utilization.gpu,"
          "temperature.gpu,power.draw,power.limit,name")


@dataclass
class GpuStat:
    name: str
    mem_used_mib: int
    mem_total_mib: int
    mem_free_mib: int
    util_pct: int
    temp_c: int
    power_draw_w: float | None = None
    power_limit_w: float | None = None


def _int(token: str) -> int:
    token = token.strip()
    try:
        return int(token)
    except ValueError:
        return 0


def _float(token: str) -> float | None:
    try:
        return float(token.strip())
    except (ValueError, AttributeError):
        return None


def parse_nvidia_smi(text: str) -> list[GpuStat]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        out.append(GpuStat(
            mem_used_mib=_int(parts[0]), mem_total_mib=_int(parts[1]),
            mem_free_mib=_int(parts[2]), util_pct=_int(parts[3]),
            temp_c=_int(parts[4]), power_draw_w=_float(parts[5]),
            power_limit_w=_float(parts[6]), name=",".join(parts[7:]).strip(),
        ))
    return out


def nvidia_smi_argv(ssh_target: str = "") -> list[str]:
    cmd = ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"]
    if not ssh_target:
        return cmd
    if not valid_ssh_target(ssh_target):
        raise ValueError(f"unsafe ssh target: {ssh_target!r}")
    return ["ssh", *SSH_OPTS, ssh_target, *cmd]


def query_gpus(ssh_target: str = "") -> list[GpuStat]:
    if not ssh_target and shutil.which("nvidia-smi") is None:
        return []
    try:
        res = subprocess.run(nvidia_smi_argv(ssh_target),
                             capture_output=True, text=True, check=False, timeout=5)
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    if res.returncode != 0:
        return []
    return parse_nvidia_smi(res.stdout)


def parse_compute_caps(text: str) -> list[str]:
    """Parse `nvidia-smi --query-gpu=compute_cap` csv lines like '12.0' into
    CMAKE_CUDA_ARCHITECTURES-style tokens like '120'. Malformed lines (e.g.
    'N/A', blanks) are skipped rather than aborting the whole parse."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            major, minor = line.split(".")
            out.append(f"{int(major)}{int(minor)}")
        except ValueError:
            continue
    return out


def compute_caps_argv(ssh_target: str = "") -> list[str]:
    cmd = ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
    if not ssh_target:
        return cmd
    if not valid_ssh_target(ssh_target):
        raise ValueError(f"unsafe ssh target: {ssh_target!r}")
    return ["ssh", *SSH_OPTS, ssh_target, *cmd]


def query_compute_caps(ssh_target: str = "") -> list[str]:
    if not ssh_target and shutil.which("nvidia-smi") is None:
        return []
    try:
        res = subprocess.run(compute_caps_argv(ssh_target),
                             capture_output=True, text=True, check=False, timeout=5)
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    if res.returncode != 0:
        return []
    return parse_compute_caps(res.stdout)
