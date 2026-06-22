import pathlib

FORBIDDEN = ("import subprocess", "import requests", "import socket",
             "from PySide6", "import PySide6")


def test_core_has_no_io_imports():
    core_dir = pathlib.Path(__file__).resolve().parents[2] / "src" / "llama_launcher" / "core"
    offenders = []
    for py in core_dir.glob("*.py"):
        text = py.read_text()
        for token in FORBIDDEN:
            if token in text:
                offenders.append(f"{py.name}: {token}")
    assert offenders == [], offenders
