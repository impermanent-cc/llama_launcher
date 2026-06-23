import llama_launcher.services.runtime as rt


class Fake:
    def __init__(self, stdout="", rc=0):
        self.stdout, self.returncode = stdout, rc


def test_logs_argv():
    assert rt.logs_argv("llama-x", "podman") == ["podman", "logs", "-f", "llama-x"]


def test_container_exists_true(monkeypatch):
    monkeypatch.setattr(rt, "_run", lambda a: Fake(rc=0))
    assert rt.container_exists("llama-x", "podman") is True


def test_container_exists_false(monkeypatch):
    monkeypatch.setattr(rt, "_run", lambda a: Fake(rc=1))
    assert rt.container_exists("llama-x", "podman") is False


def test_stats_parses_json(monkeypatch):
    monkeypatch.setattr(rt, "_run",
        lambda a: Fake(stdout='[{"CPUPerc":"12.5%","MemUsage":"1.2GB / 16GB"}]', rc=0))
    s = rt.stats("llama-x", "podman")
    assert s == {"cpu_perc": "12.5%", "mem_usage": "1.2GB / 16GB"}


def test_stats_none_on_error(monkeypatch):
    monkeypatch.setattr(rt, "_run", lambda a: Fake(stdout="", rc=125))
    assert rt.stats("llama-x", "podman") is None
