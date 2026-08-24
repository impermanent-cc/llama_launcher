import requests

from llama_launcher.services.metrics import url_host


def derive_status(container_state: str, health: str) -> str:
    """Map a container state + /health probe result to a display status.

    `health` is one of "ready" | "loading" | "down" (see `probe_health`).
    """
    if container_state == "absent":
        return "stopped"
    if container_state == "stopped":
        return "error"
    # running
    if health == "ready":
        return "running"
    if health == "loading":
        return "loading"
    return "starting"


def probe_health(port: int, timeout: float = 1.0, host: str = "127.0.0.1") -> str:
    """Probe /health, distinguishing a loading model from a dead server.

    llama-server binds its port before the model finishes loading and answers
    /health with 503 during that window; only 200 means ready. Anything else
    (other status, connection refused/timeout) is "down".
    """
    try:
        r = requests.get(f"http://{url_host(host)}:{port}/health", timeout=timeout)
    except requests.RequestException:
        return "down"
    if r.status_code == 200:
        return "ready"
    if r.status_code == 503:
        return "loading"
    return "down"
