"""Fixtures for the repository guard tests.

ALLOWLIST names files or directories (a directory covers everything beneath
it) whose non-ASCII text or dash characters are data rather than prose:
recorded tool output, multilingual samples, third-party license text. Every
entry must exist and must still contain something the guards would flag.
GUARD_REPO_ROOT overrides the repository root so the guards can run against
a repository from outside it.
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(
    os.environ.get("GUARD_REPO_ROOT") or Path(__file__).resolve().parents[2]
)

ALLOWLIST: tuple[str, ...] = (
    # User-facing documents kept as released in v0.1.0; their glyphs and
    # dashes are cleaned up in a documentation cycle.
    "README.md",
    "CHANGELOG.md",
    "RPC.md",
)

BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".svg",
    ".pdf",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".onnx",
    ".wav",
    ".ogg",
    ".mp3",
    ".zip",
    ".gz",
    ".pyc",
}
SKIP_NAMES = {"uv.lock", "poetry.lock", "package-lock.json"}


def is_allowlisted(rel: str) -> bool:
    return any(rel == e or rel.startswith(e + "/") for e in ALLOWLIST)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def tracked_files() -> list[Path]:
    """Every tracked text file outside ALLOWLIST, as absolute paths."""
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [
        REPO_ROOT / name
        for name in out.split("\0")
        if name
        and Path(name).suffix.lower() not in BINARY_SUFFIXES
        and Path(name).name not in SKIP_NAMES
        and not is_allowlisted(name)
    ]


@pytest.fixture(scope="session")
def allowlist() -> tuple[str, ...]:
    return ALLOWLIST


@pytest.fixture(scope="session")
def binary_suffixes() -> set[str]:
    return BINARY_SUFFIXES
