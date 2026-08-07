from dataclasses import dataclass


@dataclass
class PropsInfo:
    build: str | None
    n_ctx: int | None
    model_alias: str | None
    total_slots: int | None
    modalities: dict


def _first(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _as(typ, v):
    # bool is a subclass of int; exclude it when we want a real int
    if typ is int and isinstance(v, bool):
        return None
    return v if isinstance(v, typ) else None


def parse_props(data: dict) -> PropsInfo:
    """Parse a llama-server /props body. Total — never raises on bad input."""
    if not isinstance(data, dict):
        return PropsInfo(None, None, None, None, {})
    dgs = data.get("default_generation_settings")
    dgs = dgs if isinstance(dgs, dict) else {}
    mod = data.get("modalities")
    modalities = ({k: v for k, v in mod.items() if isinstance(v, bool)}
                  if isinstance(mod, dict) else {})
    return PropsInfo(
        build=_as(str, data.get("build_info")),
        n_ctx=_as(int, _first(data.get("n_ctx"), dgs.get("n_ctx"))),
        model_alias=_as(str, _first(data.get("model_alias"), dgs.get("model"))),
        total_slots=_as(int, data.get("total_slots")),
        modalities=modalities,
    )
