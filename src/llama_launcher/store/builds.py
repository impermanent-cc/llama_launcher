import json
import uuid
from pathlib import Path

from llama_launcher.store._io import write_private

from llama_launcher.core.build_command import config_slug
from llama_launcher.core.build_spec import (
    BuildConfig, BuildOutput, build_config_to_dict, build_config_from_dict,
    build_output_to_dict, build_output_from_dict,
)


def builds_dir(base_dir: Path) -> Path:
    d = base_dir / "builds"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_build_config(cfg: BuildConfig, base_dir: Path) -> Path:
    slug = config_slug(cfg.name)
    if slug == "outputs":
        # builds/outputs.json is the registry; a config named "outputs" would
        # slug onto it and silently destroy every recorded build.
        raise ValueError(
            'The name "outputs" is reserved; pick another config name.')
    path = builds_dir(base_dir) / f"{slug}.json"
    write_private(path, json.dumps(build_config_to_dict(cfg), indent=2))
    return path


def list_build_configs(base_dir: Path) -> list[BuildConfig]:
    d = builds_dir(base_dir)
    out: list[BuildConfig] = []
    for p in sorted(d.glob("*.json")):
        if p.name == "outputs.json":
            continue  # reserved registry filename, not a config
        # One truncated/corrupt config must not brick the Build tab -- skip
        # it, keep the rest. Mirrors store.profiles.list_profiles.
        try:
            out.append(build_config_from_dict(json.loads(p.read_text())))
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    return out


def delete_build_config(name: str, base_dir: Path) -> None:
    path = builds_dir(base_dir) / f"{config_slug(name)}.json"
    if path.exists():
        path.unlink()


def _outputs_path(base_dir: Path) -> Path:
    return builds_dir(base_dir) / "outputs.json"


def load_outputs(base_dir: Path) -> list[BuildOutput]:
    path = _outputs_path(base_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        # A corrupt registry must not brick the Build tab -- move it aside
        # and start fresh, mirroring store.profiles' corrupt-file guards.
        path.rename(path.with_suffix(".json.bad"))
        return []
    if not isinstance(data, list):
        path.rename(path.with_suffix(".json.bad"))
        return []
    out: list[BuildOutput] = []
    for entry in data:
        try:
            out.append(build_output_from_dict(entry))
        except (ValueError, TypeError, KeyError):
            continue
    return out


def _save_outputs(outputs: list[BuildOutput], base_dir: Path) -> None:
    data = [build_output_to_dict(o) for o in outputs]
    write_private(_outputs_path(base_dir), json.dumps(data, indent=2))


def add_output(out: BuildOutput, base_dir: Path) -> None:
    outputs = load_outputs(base_dir)
    outputs.append(out)
    _save_outputs(outputs, base_dir)


def remove_output(output_id: str, base_dir: Path) -> None:
    outputs = [o for o in load_outputs(base_dir) if o.id != output_id]
    _save_outputs(outputs, base_dir)


def write_containerfile(cfg: BuildConfig, text: str, base_dir: Path) -> Path:
    path = builds_dir(base_dir) / f"{config_slug(cfg.name)}.containerfile"
    write_private(path, text)
    return path


def new_output_id() -> str:
    return uuid.uuid4().hex[:12]
