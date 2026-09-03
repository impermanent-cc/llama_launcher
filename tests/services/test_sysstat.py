import pytest

from llama_launcher.services.sysstat import (
    CpuSampler,
    MemStat,
    cpu_percentages,
    parse_loadavg,
    parse_meminfo,
    parse_proc_stat,
    remote_meminfo_argv,
)

_STAT_A = (
    "cpu  100 0 100 800 0 0 0 0 0 0\n"
    "cpu0 50 0 50 400 0 0 0 0 0 0\n"
    "cpu1 50 0 50 400 0 0 0 0 0 0\n"
    "intr 123\n"
)
# +50 busy, +50 idle on aggregate -> 50% overall next tick
_STAT_B = (
    "cpu  124 0 126 850 0 0 0 0 0 0\n"
    "cpu0 62 0 63 425 0 0 0 0 0 0\n"
    "cpu1 62 0 63 425 0 0 0 0 0 0\n"
    "intr 200\n"
)


def test_parse_proc_stat_idle_and_total():
    d = parse_proc_stat(_STAT_A)
    assert d["cpu"] == (800, 1000)  # idle=idle+iowait=800, total=sum=1000
    assert d["cpu0"] == (400, 500) and d["cpu1"] == (400, 500)
    assert "intr" not in d


def test_parse_proc_stat_excludes_guest_from_total():
    # guest(=nums[8]) and guest_nice(=nums[9]) are already counted in user/nice,
    # so they must not be added again into total.
    line = "cpu  100 0 100 800 0 0 0 0 40 10\n"  # last two are guest, guest_nice
    idle, total = parse_proc_stat(line)["cpu"]
    assert idle == 800
    assert total == 1000  # 100+0+100+800+0+0+0+0, NOT +40+10


def test_cpu_percentages_delta():
    overall, cores = cpu_percentages(parse_proc_stat(_STAT_A), parse_proc_stat(_STAT_B))
    assert overall == 50.0
    assert cores == [50.0, 50.0]


def test_cpu_sampler_first_call_zero_then_delta():
    s = CpuSampler()
    o0, c0 = s.sample(_STAT_A)
    assert o0 == 0.0 and c0 == [0.0, 0.0]  # no previous sample yet
    o1, c1 = s.sample(_STAT_B)
    assert o1 == 50.0 and c1 == [50.0, 50.0]


def test_parse_meminfo():
    text = "MemTotal:       1000 kB\nMemFree:  100 kB\nMemAvailable:  400 kB\n"
    m = parse_meminfo(text)
    assert m == MemStat(used_bytes=(1000 - 400) * 1024, total_bytes=1000 * 1024)


def test_parse_meminfo_falls_back_to_memfree():
    m = parse_meminfo("MemTotal: 1000 kB\nMemFree: 250 kB\n")
    assert m.used_bytes == (1000 - 250) * 1024


def test_parse_loadavg():
    assert parse_loadavg("2.40 1.80 1.50 3/1234 56789") == (2.40, 1.80, 1.50)
    assert parse_loadavg("garbage") == (0.0, 0.0, 0.0)


def test_remote_meminfo_argv_builds_guarded_ssh():
    argv = remote_meminfo_argv("user@box2")
    assert argv[0] == "ssh" and "BatchMode=yes" in argv
    assert argv[-3:] == ["user@box2", "cat", "/proc/meminfo"]


def test_remote_meminfo_argv_rejects_flag_injection():
    with pytest.raises(ValueError):
        remote_meminfo_argv("-oProxyCommand=evil")
