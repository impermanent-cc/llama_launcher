"""HTTP client for the llama-server router control plane.

Two rules that are not optional:

1. Every GET carries `autoload=false`. Router GET endpoints load the model by
   default when `?model=` names an unloaded one, so a naive poll would load a
   model merely by observing it — the opposite of what an idle host is for.
2. /v1/models and /models are public even with --api-key set; everything else
   needs the key. See tools/server/server-http.cpp.
"""

import requests

from llama_launcher.core.router_events import parse_sse_event
from llama_launcher.core.router_models import RouterModel, parse_models

_NO_AUTOLOAD = {"autoload": "false"}


def base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def auth_headers(api_key: str | None) -> dict:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def list_models(host: str, port: int, api_key: str | None,
                timeout: float = 3.0) -> list[RouterModel]:
    try:
        r = requests.get(f"{base_url(host, port)}/v1/models",
                         headers=auth_headers(api_key), params=dict(_NO_AUTOLOAD),
                         timeout=timeout)
        if r.status_code != 200:
            return []
        return parse_models(r.json())
    except (requests.RequestException, ValueError):
        return []


def _post_model(host: str, port: int, api_key: str | None, path: str,
                model_id: str, timeout: float) -> bool:
    try:
        r = requests.post(f"{base_url(host, port)}{path}",
                          headers=auth_headers(api_key), json={"model": model_id},
                          timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def load_model(host: str, port: int, api_key: str | None, model_id: str,
               timeout: float = 10.0) -> bool:
    return _post_model(host, port, api_key, "/models/load", model_id, timeout)


def unload_model(host: str, port: int, api_key: str | None, model_id: str,
                 timeout: float = 10.0) -> bool:
    return _post_model(host, port, api_key, "/models/unload", model_id, timeout)


def iter_sse_events(host: str, port: int, api_key: str | None, timeout: float = 60.0):
    """Yield RouterEvents from /models/sse until the stream ends or errors."""
    with requests.get(f"{base_url(host, port)}/models/sse",
                      headers=auth_headers(api_key), stream=True,
                      timeout=timeout) as r:
        if r.status_code != 200:
            return
        block: list[str] = []
        for line in r.iter_lines(decode_unicode=True):
            if line is None:
                continue
            if line == "":
                event = parse_sse_event("\n".join(block))
                block = []
                if event is not None:
                    yield event
                continue
            block.append(line)
        if block:
            event = parse_sse_event("\n".join(block))
            if event is not None:
                yield event


def make_sse_reader(host: str, port: int, api_key: str | None):
    """A QThread that emits `event` per RouterEvent and `failed` when the stream dies.

    The caller is expected to fall back to polling /v1/models on `failed` and to
    retry with backoff: a host meant to run for weeks must not present a dead UI
    because one stream dropped.

    Qt is imported lazily so core-only test runs and the purity check never pull
    it in through this module.
    """
    from PySide6.QtCore import QThread, Signal

    class SseReader(QThread):
        event = Signal(object)
        failed = Signal(str)

        def __init__(self):
            super().__init__()
            self._stop = False

        def stop(self):
            self._stop = True

        def run(self):
            try:
                for ev in iter_sse_events(host, port, api_key):
                    if self._stop:
                        return
                    self.event.emit(ev)
            except Exception as exc:            # network, parse, teardown
                self.failed.emit(str(exc))
                return
            if not self._stop:
                self.failed.emit("stream closed")

    return SseReader()
