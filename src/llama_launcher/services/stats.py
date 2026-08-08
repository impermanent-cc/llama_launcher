from dataclasses import dataclass

from llama_launcher.services.gpu import query_gpus
from llama_launcher.services.sysstat import (
    CpuStat, MemStat, CpuSampler, parse_meminfo, parse_loadavg, read_system,
)
from llama_launcher.services.container_stats import query_container_stats


@dataclass(frozen=True)
class StatsSnapshot:
    gpus: list
    cpu: CpuStat | None
    mem: MemStat | None
    container: object | None      # ContainerStat | None
    gpu_available: bool


def build_snapshot(container_name: str, binary: str,
                   cpu_sampler: CpuSampler) -> StatsSnapshot:
    gpus = query_gpus()
    stat_text, mem_text, load_text = read_system()
    if stat_text is not None:
        overall, cores = cpu_sampler.sample(stat_text)
        cpu = CpuStat(overall_pct=overall, per_core_pct=cores,
                      load=parse_loadavg(load_text))
        mem = parse_meminfo(mem_text)
    else:
        cpu, mem = None, None
    container = query_container_stats(container_name, binary) if container_name else None
    return StatsSnapshot(gpus=gpus, cpu=cpu, mem=mem, container=container,
                         gpu_available=bool(gpus))
