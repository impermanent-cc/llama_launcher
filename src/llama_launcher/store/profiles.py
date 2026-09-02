import json
import os
from dataclasses import asdict
from pathlib import Path

from llama_launcher.store._io import write_private

from llama_launcher.core.spec import (
    Profile, Mount, LoraRef, Runtime, RouterMember, RpcWorker, slugify,
)


def default_base_dir() -> Path:
    root = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(root) / "llama-launcher"


def profile_to_dict(p: Profile) -> dict:
    return asdict(p)


# The container runtime is used as argv[0] of every launch; a loaded profile
# must not be able to name an arbitrary executable there. The GUI combo
# constrains it; this guards the headless/loaded-JSON path.
_ALLOWED_BINARIES = ("podman", "docker")


def profile_from_dict(d: dict) -> Profile:
    rt = dict(d.get("runtime", {}))
    if rt.get("binary") not in _ALLOWED_BINARIES:
        rt["binary"] = "podman"
    rt["rpc_workers"] = [RpcWorker(**w) for w in rt.get("rpc_workers", [])]
    return Profile(
        name=d["name"],
        image=d.get("image", ""),
        runtime=Runtime(**rt),
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
    # A profile may hold a cleartext api-key; keep the directory owner-only.
    # mkdir's mode only applies on creation, so tighten a pre-existing dir too.
    try:
        d.chmod(0o700)
    except OSError:
        pass
    return d


def save_profile(p: Profile, base_dir: Path) -> Path:
    path = _profiles_dir(base_dir) / f"{slugify(p.name)}.json"
    data = json.dumps(profile_to_dict(p), indent=2)
    # Write owner-only from the start: a profile can carry an api-key, and
    # write_text() + a later chmod would leave it world-readable in between.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(data)
    os.chmod(path, 0o600)          # tighten a pre-existing wider file too
    return path


def load_profile(path: Path) -> Profile:
    return profile_from_dict(json.loads(Path(path).read_text()))


def list_profiles(base_dir: Path) -> list[Profile]:
    d = _profiles_dir(base_dir)
    out: list[Profile] = []
    for p in sorted(d.glob("*.json")):
        # One truncated/corrupt profile (a killed write, a full disk, a bad hand
        # edit) must not brick the whole GUI at startup -- skip it, keep the rest.
        try:
            out.append(load_profile(p))
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return out


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
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}                       # a corrupt config must not crash startup
    return data if isinstance(data, dict) else {}


def save_config(cfg: dict, base_dir: Path) -> None:
    write_private(_config_path(base_dir), json.dumps(cfg, indent=2))
