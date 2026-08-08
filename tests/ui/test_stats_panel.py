from llama_launcher.ui.panels.stats_panel import StatsPanel
from llama_launcher.services.stats import StatsSnapshot
from llama_launcher.services.sysstat import CpuStat, MemStat
from llama_launcher.services.gpu import GpuStat
from llama_launcher.services.container_stats import ContainerStat


def _full_snapshot():
    return StatsSnapshot(
        gpus=[GpuStat(name="RTX 5080", mem_used_mib=14746, mem_total_mib=16384,
                      mem_free_mib=1638, util_pct=78, temp_c=63, power_draw_w=210.0,
                      power_limit_w=360.0)],
        cpu=CpuStat(overall_pct=46.0, per_core_pct=[40.0, 52.0], load=(2.4, 1.8, 1.5)),
        mem=MemStat(used_bytes=40 * 1024**3, total_bytes=64 * 1024**3),
        container=ContainerStat(name="llama-qwen", cpu_pct=180.0,
                                mem_used_bytes=4 * 1024**3, mem_limit_bytes=64 * 1024**3),
        gpu_available=True)


def test_renders_all_sections(qtbot):
    p = StatsPanel(); qtbot.addWidget(p)
    p.update_stats(_full_snapshot())
    assert "RTX 5080" in p.gpu_label.text() and "78%" in p.gpu_label.text()
    assert "210" in p.gpu_label.text()                 # power draw
    assert "46" in p.system_label.text() and "2.4" in p.system_label.text()
    assert "llama-qwen" in p.container_label.text() and "180" in p.container_label.text()


def test_gpu_unavailable(qtbot):
    p = StatsPanel(); qtbot.addWidget(p)
    snap = StatsSnapshot(gpus=[], cpu=None, mem=None, container=None, gpu_available=False)
    p.update_stats(snap)
    assert "unavailable" in p.gpu_label.text().lower()
    assert "unavailable" in p.system_label.text().lower()
    assert "no server" in p.container_label.text().lower()


def test_update_twice_grows_sparkline_without_error(qtbot):
    p = StatsPanel(); qtbot.addWidget(p)
    p.update_stats(_full_snapshot())
    p.update_stats(_full_snapshot())               # second sample extends history
    assert "RTX 5080" in p.gpu_label.text()


def test_panel_width_constant_across_updates(qtbot):
    # Regression: the sparkline grew 1->N chars as history filled, widening the
    # labels and making the dock/window grow every tick. Width must be constant
    # from the first update, whatever the values.
    p = StatsPanel(); qtbot.addWidget(p)
    p.update_stats(_full_snapshot())
    w0 = p.gpu_label.sizeHint().width()
    s0 = p.system_label.sizeHint().width()
    # feed many varied snapshots
    from llama_launcher.services.stats import StatsSnapshot
    from llama_launcher.services.sysstat import CpuStat, MemStat
    from llama_launcher.services.gpu import GpuStat
    from llama_launcher.services.container_stats import ContainerStat
    for u in range(0, 101, 3):
        p.update_stats(StatsSnapshot(
            gpus=[GpuStat(name="RTX 5080", mem_used_mib=14746, mem_total_mib=16384,
                          mem_free_mib=1638, util_pct=u, temp_c=63, power_draw_w=210.0,
                          power_limit_w=360.0)],
            cpu=CpuStat(overall_pct=float(u), per_core_pct=[float(u)] * 8, load=(2.4, 1.8, 1.5)),
            mem=MemStat(used_bytes=40 * 1024**3, total_bytes=64 * 1024**3),
            container=ContainerStat(name="llama-e2b", cpu_pct=float(u * 2),
                                    mem_used_bytes=4 * 1024**3, mem_limit_bytes=64 * 1024**3),
            gpu_available=True))
    assert p.gpu_label.sizeHint().width() <= w0        # never grows
    assert p.system_label.sizeHint().width() <= s0
