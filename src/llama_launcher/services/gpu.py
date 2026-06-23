import shutil
import subprocess
from dataclasses import dataclass

_QUERY = "memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu,name"


@dataclass
class GpuStat:
    name: str
    mem_used_mib: int
    mem_total_mib: int
    mem_free_mib: int
    util_pct: int
    temp_c: int


def _int(token: str) -> int:
    token = token.strip()
    try:
        return int(token)
    except ValueError:
        return 0


def parse_nvidia_smi(text: str) -> list[GpuStat]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        out.append(GpuStat(
            mem_used_mib=_int(parts[0]), mem_total_mib=_int(parts[1]),
            mem_free_mib=_int(parts[2]), util_pct=_int(parts[3]),
            temp_c=_int(parts[4]), name=",".join(parts[5:]).strip(),
        ))
    return out


def query_gpus() -> list[GpuStat]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        res = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    if res.returncode != 0:
        return []
    return parse_nvidia_smi(res.stdout)


def free_vram_bytes() -> int | None:
    gpus = query_gpus()
    if not gpus:
        return None
    return max(g.mem_free_mib for g in gpus) * 1024 * 1024
