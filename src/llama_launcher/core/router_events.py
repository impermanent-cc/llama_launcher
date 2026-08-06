"""Parse llama-server router `GET /models/sse` event frames.

Pure module: no IO, no Qt. The caller is responsible for splitting the stream
into blank-line-separated blocks and handing each block here.
"""

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RouterEvent:
    model: str
    event: str
    data: dict = field(default_factory=dict)


def parse_sse_event(block: str) -> RouterEvent | None:
    """Return the event described by one SSE block, or None if it isn't one.

    Comment pings (lines starting with ':') and malformed JSON yield None so a
    keep-alive never looks like a state change.
    """
    payload_lines = [
        line[len("data:"):].strip()
        for line in block.splitlines()
        if line.startswith("data:")
    ]
    if not payload_lines:
        return None

    try:
        obj = json.loads("".join(payload_lines))
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None

    event = obj.get("event")
    if not isinstance(event, str) or not event:
        return None

    data = obj.get("data")
    return RouterEvent(
        model=obj.get("model") or "",
        event=event,
        data=data if isinstance(data, dict) else {},
    )
