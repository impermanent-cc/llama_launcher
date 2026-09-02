"""Docker/podman parity for the runtime layer.

The launcher advertises docker as a first-class runtime, so no helper may
assume podman-only CLI shapes.
"""
from unittest.mock import patch

import llama_launcher.services.runtime as rt


class Fake:
    def __init__(self, stdout="", rc=0):
        self.stdout, self.returncode = stdout, rc


# -- ps output: podman emits a JSON array, docker emits NDJSON ---------------

def test_parse_ps_json_accepts_podman_array():
    out = '[{"Names":["llama-x"],"State":"running","Labels":{"llama-launcher.profile":"x"}}]'
    rows = rt.parse_ps_json(out)
    assert rows == [{"name": "llama-x", "running": True, "profile": "x", "mode": "server"}]


def test_parse_ps_json_accepts_docker_ndjson():
    # docker `ps --format json` prints one bare JSON object per line (NDJSON),
    # with a bare-string Names and a "Labels" that may be a comma string.
    out = ('{"Names":"llama-a","State":"running","Labels":"llama-launcher.profile=a"}\n'
           '{"Names":"llama-b","State":"exited","Labels":"llama-launcher.profile=b"}\n')
    rows = {r["name"]: r for r in rt.parse_ps_json(out)}
    assert rows["llama-a"]["running"] is True and rows["llama-a"]["profile"] == "a"
    assert rows["llama-b"]["running"] is False and rows["llama-b"]["profile"] == "b"


def test_parse_ps_json_docker_single_object_no_newline():
    out = '{"Names":"llama-solo","State":"running","Labels":""}'
    rows = rt.parse_ps_json(out)
    assert rows == [{"name": "llama-solo", "running": True, "profile": "solo", "mode": "server"}]


# -- container_exists / image_exists: inspect, not podman-only subcommands ----

def test_container_exists_uses_inspect():
    seen = {}
    def fake_run(args, timeout=10):
        seen["args"] = args
        return Fake(rc=0)
    with patch.object(rt, "_run", fake_run):
        assert rt.container_exists("llama-x", "docker") is True
    assert "inspect" in seen["args"] and "exists" not in seen["args"]
    assert seen["args"][0] == "docker"


def test_image_exists_uses_inspect():
    seen = {}
    def fake_run(args, timeout=10):
        seen["args"] = args
        return Fake(rc=1)
    with patch.object(rt, "_run", fake_run):
        assert rt.image_exists("img", "docker") is False
    assert "inspect" in seen["args"] and "exists" not in seen["args"]


# -- stats: must read podman's key names too (the default runtime) -----------

def test_stats_reads_podman_keys():
    with patch.object(rt, "_run",
                      lambda a: Fake(stdout='[{"cpu_percent":"7.0%","mem_usage":"1GiB / 8GiB"}]', rc=0)):
        s = rt.stats("llama-x", "podman")
    assert s == {"cpu_perc": "7.0%", "mem_usage": "1GiB / 8GiB"}


def test_stats_still_reads_docker_keys():
    with patch.object(rt, "_run",
                      lambda a: Fake(stdout='{"CPUPerc":"9%","MemUsage":"2GB / 16GB"}', rc=0)):
        s = rt.stats("llama-x", "docker")
    assert s == {"cpu_perc": "9%", "mem_usage": "2GB / 16GB"}


# -- remote transport: podman uses --connection, docker uses --context -------

def test_base_uses_context_for_docker():
    assert rt._base("docker", "box-b") == ["docker", "--context", "box-b"]
    assert rt._base("podman", "box-b") == ["podman", "--connection", "box-b"]
    assert rt._base("docker") == ["docker"]


def test_connection_add_argv_docker_uses_context_create():
    assert rt.connection_add_argv("box-b", "me@10.0.0.2:22", "docker") == \
        ["docker", "context", "create", "box-b", "--docker", "host=ssh://me@10.0.0.2:22"]


def test_connection_remove_argv_docker_uses_context_rm():
    assert rt.connection_remove_argv("box-b", "docker") == \
        ["docker", "context", "rm", "box-b"]
