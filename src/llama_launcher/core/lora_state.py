"""Parse llama-server `GET /lora-adapters` payloads.

Pure module: no IO, no Qt.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LoraAdapter:
    id: int
    path: str = ""
    scale: float = 0.0

    @property
    def active(self) -> bool:
        """Whether this adapter currently contributes anything.

        A loaded-but-zeroed adapter is the normal resting state under
        `--lora-init-without-apply`, so "loaded" and "active" are genuinely
        different questions and the UI asks both.
        """
        return self.scale != 0.0

    @property
    def name(self) -> str:
        """Filename for display; the full path is the tooltip's job."""
        return self.path.rsplit("/", 1)[-1] or str(self.id)


def parse_adapters(payload) -> list[LoraAdapter]:
    """Adapters from a /lora-adapters body, skipping anything malformed.

    The payload is a bare JSON array, not an object with a "data" key like the
    router's /v1/models. An adapter with no usable integer id is dropped rather
    than defaulted: ids are what a POST addresses, so inventing one would let
    the UI rescale the wrong adapter.
    """
    if not isinstance(payload, list):
        return []

    out: list[LoraAdapter] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("id")
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            continue
        raw_scale = item.get("scale")
        scale = float(raw_scale) if isinstance(raw_scale, (int, float)) \
            and not isinstance(raw_scale, bool) else 0.0
        path = item.get("path")
        out.append(LoraAdapter(
            id=raw_id,
            path=path if isinstance(path, str) else "",
            scale=scale,
        ))
    return out
