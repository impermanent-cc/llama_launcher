import json
from pathlib import Path

from llama_launcher.core.presets import Preset
from llama_launcher.core.spec import slugify
from llama_launcher.store._io import write_private


def preset_to_dict(p: Preset) -> dict:
    return {"key": p.key, "label": p.label,
            "settings": dict(p.settings), "source": p.source}


def preset_from_dict(d: dict) -> Preset:
    return Preset(
        key=d["key"],
        label=d.get("label", d["key"]),
        settings=dict(d.get("settings", {})),
        source=d.get("source", "user"),      # stored presets are user presets
    )


def _presets_dir(base_dir: Path) -> Path:
    d = base_dir / "presets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_preset(p: Preset, base_dir: Path) -> Path:
    path = _presets_dir(base_dir) / f"{slugify(p.key)}.json"
    write_private(path, json.dumps(preset_to_dict(p), indent=2))
    return path


def list_presets(base_dir: Path) -> list[Preset]:
    d = _presets_dir(base_dir)
    presets = []
    for p in sorted(d.glob("*.json")):
        try:
            presets.append(preset_from_dict(json.loads(p.read_text())))
        except (OSError, ValueError, KeyError, TypeError):
            continue  # skip unreadable/malformed/incomplete preset file
    return presets


def delete_preset(key: str, base_dir: Path) -> None:
    path = _presets_dir(base_dir) / f"{slugify(key)}.json"
    if path.exists():
        path.unlink()
