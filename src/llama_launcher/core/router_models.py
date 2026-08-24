"""Parse llama-server router `GET /v1/models` payloads.

Pure module: no IO, no Qt.
"""

from dataclasses import dataclass

STATUSES = ("unloaded", "loading", "loaded", "sleeping", "downloading")


@dataclass(frozen=True)
class RouterModel:
    id: str
    path: str = ""
    status: str = "unloaded"
    progress: float | None = None       # 0.0-1.0 while loading/downloading
    args: tuple = ()
    failed: bool = False
    exit_code: int | None = None


def _progress(status: dict) -> float | None:
    raw = status.get("progress")
    if not isinstance(raw, dict):
        return None

    # Load progress: {"stages": [...], "current": ..., "value": 0.5}
    value = raw.get("value")
    if isinstance(value, (int, float)):
        return float(value)

    # Download progress: {url: {"done": N, "total": M}, ...}; possibly parallel.
    done = total = 0
    for entry in raw.values():
        if isinstance(entry, dict):
            done += entry.get("done") or 0
            total += entry.get("total") or 0
    if total <= 0:
        return None
    return done / total


def parse_models(payload: dict) -> list[RouterModel]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []

    out: list[RouterModel] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        status = item.get("status")
        status = status if isinstance(status, dict) else {}
        args = status.get("args")
        exit_code = status.get("exit_code")
        out.append(RouterModel(
            id=model_id,
            path=item.get("path") or "",
            status=status.get("value") or "unloaded",
            progress=_progress(status),
            args=tuple(args) if isinstance(args, list) else (),
            failed=bool(status.get("failed")),
            exit_code=exit_code if isinstance(exit_code, int) else None,
        ))
    return out
