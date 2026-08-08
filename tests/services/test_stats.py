import llama_launcher.services.stats as stats
from llama_launcher.services.sysstat import CpuSampler
from llama_launcher.services.gpu import GpuStat
from llama_launcher.services.container_stats import ContainerStat


def _gpu():
    return GpuStat(name="G", mem_used_mib=1, mem_total_mib=2, mem_free_mib=1,
                   util_pct=10, temp_c=40)


def test_build_snapshot_assembles_all(monkeypatch):
    monkeypatch.setattr(stats, "query_gpus", lambda: [_gpu()])
    monkeypatch.setattr(stats, "read_system",
                        lambda: ("cpu  1 0 1 8 0 0 0 0 0 0\n", "MemTotal: 1000 kB\n", "1 2 3 x y"))
    monkeypatch.setattr(stats, "query_container_stats",
                        lambda name, binary: ContainerStat("c", 5.0, 10, 20))
    snap = stats.build_snapshot("c", "podman", CpuSampler())
    assert snap.gpu_available is True and len(snap.gpus) == 1
    assert snap.cpu is not None and snap.cpu.load == (1.0, 2.0, 3.0)
    assert snap.mem is not None and snap.mem.total_bytes == 1000 * 1024
    assert snap.container.name == "c"


def test_build_snapshot_degrades(monkeypatch):
    monkeypatch.setattr(stats, "query_gpus", lambda: [])
    monkeypatch.setattr(stats, "read_system", lambda: (None, None, None))
    monkeypatch.setattr(stats, "query_container_stats", lambda name, binary: None)
    snap = stats.build_snapshot("", "podman", CpuSampler())
    assert snap.gpu_available is False and snap.gpus == []
    assert snap.cpu is None and snap.mem is None and snap.container is None
