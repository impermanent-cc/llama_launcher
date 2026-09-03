from llama_launcher.services.container_stats import (
    ContainerStat,
    parse_container_stats,
    parse_size,
)

_PODMAN = (
    '[{"id":"070c477a2980","name":"llama-qwen","cpu_percent":"18.5%",'
    '"mem_usage":"4.2GB / 25.2GB","mem_percent":"16.7%","pids":"12"}]'
)
# docker-style keys (the app also supports docker)
_DOCKER = '[{"Name":"llama-qwen","CPUPerc":"18.5%","MemUsage":"4.2GiB / 24GiB"}]'


def test_parse_size_decimal_and_binary():
    assert parse_size("4.2GB") == int(4.2 * 10**9)
    assert parse_size("204.8kB") == int(204.8 * 10**3)
    assert parse_size("4.2GiB") == int(4.2 * 1024**3)
    assert parse_size("512MiB") == 512 * 1024**2
    assert parse_size("") is None


def test_parse_container_stats_podman():
    s = parse_container_stats(_PODMAN)
    assert s == ContainerStat(
        name="llama-qwen",
        cpu_pct=18.5,
        mem_used_bytes=int(4.2 * 10**9),
        mem_limit_bytes=int(25.2 * 10**9),
    )


def test_parse_container_stats_docker_keys():
    s = parse_container_stats(_DOCKER)
    assert s.name == "llama-qwen" and s.cpu_pct == 18.5
    assert s.mem_used_bytes == int(4.2 * 1024**3)


def test_parse_container_stats_empty_and_bad():
    assert parse_container_stats("[]") is None
    assert parse_container_stats("not json") is None
    assert parse_container_stats("{}") is None
