import json
import os
from dataclasses import asdict
from pathlib import Path

from llama_launcher.core.spec import (
    Profile, Mount, LoraRef, Runtime, RouterMember, slugify,
)


def default_base_dir() -> Path:
    root = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(root) / "llama-launcher"


def profile_to_dict(p: Profile) -> dict:
    return asdict(p)


def profile_from_dict(d: dict) -> Profile:
    return Profile(
        name=d["name"],
        image=d.get("image", ""),
        runtime=Runtime(**d.get("runtime", {})),
        mounts=[Mount(**m) for m in d.get("mounts", [])],
        model=d.get("model", ""),
        mmproj=d.get("mmproj"),
        draft_model=d.get("draft_model"),
        loras=[LoraRef(**l) for l in d.get("loras", [])],
        settings=dict(d.get("settings", {})),
        raw_args=d.get("raw_args", ""),
        mode=d.get("mode", "server"),
        members=[RouterMember(**m) for m in d.get("members", [])],
    )


def _profiles_dir(base_dir: Path) -> Path:
    d = base_dir / "profiles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_profile(p: Profile, base_dir: Path) -> Path:
    path = _profiles_dir(base_dir) / f"{slugify(p.name)}.json"
    path.write_text(json.dumps(profile_to_dict(p), indent=2))
    return path


def load_profile(path: Path) -> Profile:
    return profile_from_dict(json.loads(Path(path).read_text()))


def list_profiles(base_dir: Path) -> list[Profile]:
    d = _profiles_dir(base_dir)
    return [load_profile(p) for p in sorted(d.glob("*.json"))]


def resolve_member_pairs(members, base_dir):
    """(RouterMember, member Profile) pairs for members whose profile exists.

    Preserves member order; silently drops members whose referenced profile is
    gone (the router preset simply omits them).
    """
    by_name = {p.name: p for p in list_profiles(base_dir)}
    return [(m, by_name[m.profile]) for m in members if m.profile in by_name]


def delete_profile(name: str, base_dir: Path) -> None:
    path = _profiles_dir(base_dir) / f"{slugify(name)}.json"
    if path.exists():
        path.unlink()


def _config_path(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "config.json"


def load_config(base_dir: Path) -> dict:
    path = _config_path(base_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_config(cfg: dict, base_dir: Path) -> None:
    _config_path(base_dir).write_text(json.dumps(cfg, indent=2))
