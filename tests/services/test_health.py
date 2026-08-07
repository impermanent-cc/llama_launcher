import requests

from llama_launcher.services import health
from llama_launcher.services.health import derive_status, probe_health


def test_absent_is_stopped():
    assert derive_status("absent", "down") == "stopped"


def test_running_container_not_answering_is_starting():
    assert derive_status("running", "down") == "starting"


def test_running_and_ready_is_running():
    assert derive_status("running", "ready") == "running"


def test_running_but_loading_is_loading():
    # llama-server binds its port and answers /health with 503 while the model
    # loads; that is distinct from the process not answering at all.
    assert derive_status("running", "loading") == "loading"


def test_stopped_container_is_error_when_expected_up():
    # container exists but is stopped while we expected it running
    assert derive_status("stopped", "down") == "error"


class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code


def test_probe_health_ready_on_200(monkeypatch):
    monkeypatch.setattr(health.requests, "get", lambda *a, **k: _Resp(200))
    assert probe_health(8080) == "ready"


def test_probe_health_loading_on_503(monkeypatch):
    monkeypatch.setattr(health.requests, "get", lambda *a, **k: _Resp(503))
    assert probe_health(8080) == "loading"


def test_probe_health_down_on_other_status(monkeypatch):
    monkeypatch.setattr(health.requests, "get", lambda *a, **k: _Resp(500))
    assert probe_health(8080) == "down"


def test_probe_health_down_on_connection_error(monkeypatch):
    def boom(*a, **k):
        raise requests.RequestException("refused")
    monkeypatch.setattr(health.requests, "get", boom)
    assert probe_health(8080) == "down"
