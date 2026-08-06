"""Launcher-managed router files: the API key and the preset INI.

The key lives in a 0600 file rather than on the command line, because argv is
visible in `podman inspect`, `ps`, the exported .sh and the diagnostic report.
llama-server reads it with --api-key-file.
"""

import secrets
from pathlib import Path

from llama_launcher.core.spec import slugify

_KEY_FILE = "api-key"
_PRESET_FILE = "models.ini"


def router_dir(base_dir: Path, router_name: str) -> Path:
    """Per-router config directory, created if missing."""
    d = Path(base_dir) / "router" / slugify(router_name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def generate_key() -> str:
    """A fresh API key, `sk-` prefixed so OpenAI-style clients accept it."""
    return "sk-" + secrets.token_urlsafe(32)


def _key_path(base_dir: Path, router_name: str) -> Path:
    return router_dir(base_dir, router_name) / _KEY_FILE


def read_api_key(base_dir: Path, router_name: str) -> str | None:
    """The stored key, or None when there isn't one yet or the file is unreadable."""
    path = _key_path(base_dir, router_name)
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
    path.write_text(key + "\n")
    path.chmod(0o600)
    return key


def ensure_api_key(base_dir: Path, router_name: str) -> str:
    """The stored key, generating and persisting one on first use."""
    existing = read_api_key(base_dir, router_name)
    if existing:
        return existing
    return _write_key(base_dir, router_name, generate_key())


def regenerate_api_key(base_dir: Path, router_name: str) -> str:
    """Replace the key. Any harness configured with the old one stops working."""
    return _write_key(base_dir, router_name, generate_key())


def write_preset(base_dir: Path, router_name: str, text: str) -> Path:
    """Write the rendered models.ini for this router and return its path."""
    path = router_dir(base_dir, router_name) / _PRESET_FILE
    path.write_text(text)
    return path
