import requests


def derive_status(container_state: str, health_ok: bool) -> str:
    if container_state == "absent":
        return "stopped"
    if container_state == "stopped":
        return "error"
    # running
    return "running" if health_ok else "starting"


def health_ok(port: int, timeout: float = 1.0, host: str = "127.0.0.1") -> bool:
    try:
        r = requests.get(f"http://{host}:{port}/health", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False
