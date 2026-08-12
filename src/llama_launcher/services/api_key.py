"""Launcher-managed router files: the API key and the preset INI.

The key lives in a 0600 file rather than on the command line, because argv is
visible in `podman inspect`, `ps`, the exported .sh and the diagnostic report.
llama-server reads it with --api-key-file.
"""

import os
import secrets
from pathlib import Path

from llama_launcher.core.spec import Profile, slugify

_KEY_FILE = "api-key"
_PRESET_FILE = "models.ini"
_GLOBAL_KEY_FILE = "global-api-key"


def router_dir(base_dir: Path, router_name: str, create: bool = True) -> Path:
    """Per-router config directory.

    Created 0700 when `create` — it holds an API key, and a group-writable
    directory would let another local user swap the key or repoint the preset,
    which the 0600 on the file itself does nothing to prevent. Readers pass
    create=False so a poll never has filesystem side effects.
    """
    d = Path(base_dir) / "router" / slugify(router_name)
    if create:
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
    return d


def generate_key() -> str:
    """A fresh API key, `sk-` prefixed so OpenAI-style clients accept it."""
    return "sk-" + secrets.token_urlsafe(32)


def _key_path(base_dir: Path, router_name: str, create: bool = True) -> Path:
    return router_dir(base_dir, router_name, create=create) / _KEY_FILE


def read_api_key(base_dir: Path, router_name: str) -> str | None:
    """The stored key, or None when there isn't one yet or the file is unreadable."""
    path = _key_path(base_dir, router_name, create=False)
    try:
        text = path.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return None


def _write_key(base_dir: Path, router_name: str, key: str) -> str:
    path = _key_path(base_dir, router_name)
    # Create with the mode already set. write_text() + chmod() leaves the secret
    # on disk world-readable (0644 under a typical umask) for the window between
    # the two calls; an end-state permission assertion cannot see that.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(key + "\n")
    path.chmod(0o600)          # in case the file already existed with a wider mode
    return key


def ensure_api_key(base_dir: Path, router_name: str) -> str:
    """The stored key, generating and persisting one on first use."""
    existing = read_api_key(base_dir, router_name)
    if existing:
        return existing
    return _write_key(base_dir, router_name, generate_key())


def write_preset(base_dir: Path, router_name: str, text: str) -> Path:
    """Write the rendered models.ini for this router and return its path."""
    path = router_dir(base_dir, router_name) / _PRESET_FILE
    path.write_text(text)
    return path


def normalize_key(raw: str) -> str:
    """Trim to the intended single-line secret; reject empty/multi-line input.

    Permissive on format on purpose -- llama-server reads the string verbatim;
    the UI (not this function) warns about a missing `sk-` prefix.
    """
    key = raw.strip()
    if not key or "\n" in key:
        raise ValueError("API key must be a non-empty, single-line string")
    return key


def global_key_path(base_dir: Path) -> Path:
    return Path(base_dir) / "router" / _GLOBAL_KEY_FILE


def read_global_key(base_dir: Path) -> str | None:
    """The shared key, or None when unset/unreadable."""
    try:
        text = global_key_path(base_dir).read_text()
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return None


def write_global_key(base_dir: Path, key: str) -> str:
    """Persist the shared key 0600, creating base_dir/router 0700."""
    path = global_key_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)   # in case it already existed wider (mkdir -p leaves
                                # an intermediate dir made by router_dir() at ~0755)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(key + "\n")
    path.chmod(0o600)
    return key


def set_profile_key(base_dir: Path, router_name: str, key: str) -> str:
    """Write a user-supplied value to the per-profile key file (0600)."""
    return _write_key(base_dir, router_name, key)


def resolve_api_key(base_dir: Path, profile: Profile) -> str | None:
    """The effective key for this profile, read-only (no generate, no write).

    own    -> the per-profile key file.
    global -> the shared global key if set, else the per-profile file (legacy).
    """
    name = profile.name
    if profile.runtime.router_key_mode == "own":
        return read_api_key(base_dir, name)
    return read_global_key(base_dir) or read_api_key(base_dir, name)


def prepare_launch_key(base_dir: Path, profile: Profile) -> str:
    """Ensure an effective key exists and write it into the per-profile file the
    container serves (--api-key-file). Returns the key.

    In global mode this copies the shared key into the per-profile file so the
    launched container serves the shared key without any command_builder change.
    """
    name = profile.name
    if profile.runtime.router_key_mode == "own":
        return ensure_api_key(base_dir, name)
    key = read_global_key(base_dir)
    if key is None:
        return ensure_api_key(base_dir, name)   # legacy: generate + persist per-profile
    return set_profile_key(base_dir, name, key)
