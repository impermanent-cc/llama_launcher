import json
from dataclasses import asdict
from pathlib import Path

from llama_launcher.core.spec import slugify
from llama_launcher.store._io import write_private


def sweep_path(base_dir, profile_name) -> Path:
    return Path(base_dir) / "sweeps" / f"{slugify(profile_name)}.json"


def save(base_dir, profile_name, sweep) -> None:
    """Write the latest sweep for a profile, replacing any prior one."""
    path = sweep_path(base_dir, profile_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_private(path, json.dumps(asdict(sweep), indent=2))


def load(base_dir, profile_name):
    """The stored sweep as a dict, or None when absent or unreadable."""
    try:
        data = json.loads(sweep_path(base_dir, profile_name).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
