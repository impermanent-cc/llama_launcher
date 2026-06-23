import pathlib

FORBIDDEN = (
    "import subprocess", "import requests", "import socket",
    "from PySide6", "import PySide6",
    "open(", ".read_text", ".read_bytes", ".stat(", "import os", "import pathlib",
)

# Modules that are pure-core (no I/O) and must be checked.
_ENHANCEMENT_MODULES = {"gguf.py", "vram.py", "prometheus.py", "report.py"}


def test_core_has_no_io_imports():
    core_dir = pathlib.Path(__file__).resolve().parents[2] / "src" / "llama_launcher" / "core"
    offenders = []
    for py in core_dir.glob("*.py"):
        text = py.read_text()
        for token in FORBIDDEN:
            if token in text:
                offenders.append(f"{py.name}: {token}")
    assert offenders == [], offenders


def test_enhancement_modules_are_pure():
    """The 4 core enhancement modules must not contain any file-read or I/O tokens."""
    core_dir = pathlib.Path(__file__).resolve().parents[2] / "src" / "llama_launcher" / "core"
    offenders = []
    for py in core_dir.glob("*.py"):
        if py.name not in _ENHANCEMENT_MODULES:
            continue
        text = py.read_text()
        for token in FORBIDDEN:
            if token in text:
                offenders.append(f"{py.name}: {token!r}")
    assert offenders == [], offenders
