from dataclasses import dataclass


@dataclass(frozen=True)
class CpuStat:
    overall_pct: float
    per_core_pct: list
    load: tuple


@dataclass(frozen=True)
class MemStat:
    used_bytes: int
    total_bytes: int


def parse_proc_stat(text: str) -> dict:
    """Map each 'cpu'/'cpuN' line to (idle_jiffies, total_jiffies).

    Fields after the label are: user nice system idle iowait irq softirq steal
    guest guest_nice. idle time counts idle + iowait.
    """
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts or not parts[0].startswith("cpu"):
            continue
        try:
            nums = [int(x) for x in parts[1:]]
        except ValueError:
            continue
        if len(nums) < 5:
            continue
        idle = nums[3] + nums[4]
        out[parts[0]] = (idle, sum(nums))
    return out


def _pct(prev, cur) -> float:
    total_d = cur[1] - prev[1]
    if total_d <= 0:
        return 0.0
    busy_d = total_d - (cur[0] - prev[0])
    return round(100.0 * busy_d / total_d, 1)


def cpu_percentages(prev: dict, cur: dict) -> tuple:
    overall = _pct(prev["cpu"], cur["cpu"]) if "cpu" in prev and "cpu" in cur else 0.0
    cores, i = [], 0
    while f"cpu{i}" in cur and f"cpu{i}" in prev:
        cores.append(_pct(prev[f"cpu{i}"], cur[f"cpu{i}"]))
        i += 1
    return overall, cores


class CpuSampler:
    """Holds the previous /proc/stat read; each sample() returns the % since it."""
    def __init__(self):
        self._prev = None

    def sample(self, proc_stat_text: str) -> tuple:
        cur = parse_proc_stat(proc_stat_text)
        if self._prev is None:
            self._prev = cur
            n = sum(1 for k in cur if k.startswith("cpu") and k != "cpu")
            return 0.0, [0.0] * n
        overall, cores = cpu_percentages(self._prev, cur)
        self._prev = cur
        return overall, cores


def parse_meminfo(text: str) -> MemStat:
    vals = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        toks = rest.split()
        if toks and toks[0].isdigit():
            vals[key.strip()] = int(toks[0]) * 1024      # kB -> bytes
    total = vals.get("MemTotal", 0)
    avail = vals.get("MemAvailable", vals.get("MemFree", 0))
    return MemStat(used_bytes=max(0, total - avail), total_bytes=total)


def parse_loadavg(text: str) -> tuple:
    toks = text.split()
    try:
        return (float(toks[0]), float(toks[1]), float(toks[2]))
    except (IndexError, ValueError):
        return (0.0, 0.0, 0.0)


def read_system() -> tuple:
    """(/proc/stat, /proc/meminfo, /proc/loadavg) texts, or (None, None, None)
    when /proc is unavailable (non-Linux)."""
    try:
        with open("/proc/stat") as f:
            stat = f.read()
        with open("/proc/meminfo") as f:
            mem = f.read()
        with open("/proc/loadavg") as f:
            load = f.read()
        return stat, mem, load
    except OSError:
        return None, None, None
