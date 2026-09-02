"""Integration tests for scripts/install-desktop.sh.

These run the real script against a temporary XDG home, with no-op stubs on PATH
for the desktop/icon-cache refresh tools so the test stays hermetic and fast.
"""
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "install-desktop.sh"
SRC_SVG = REPO / "assets" / "llama-launcher.svg"


def _stub_dir(tmp_path: Path) -> Path:
    d = tmp_path / "stubs"
    d.mkdir()
    for name in ("gtk-update-icon-cache", "gtk4-update-icon-cache",
                 "update-desktop-database", "kbuildsycoca6", "kbuildsycoca5"):
        p = d / name
        p.write_text("#!/usr/bin/env bash\nexit 0\n")
        p.chmod(0o755)
    return d


def _env(tmp_path: Path) -> dict:
    env = dict(os.environ)
    env["HOME"] = str(tmp_path / "home")
    env["XDG_DATA_HOME"] = str(tmp_path / "data")
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    env["PATH"] = f"{_stub_dir(tmp_path)}:{env['PATH']}"
    # Point the script at a stub interpreter so the test doesn't depend on a
    # real .venv in the repo root (absent in CI). The script only checks that
    # it's an executable file; the desktop entry's Exec line isn't asserted.
    venv_py = tmp_path / "venv" / "bin" / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("#!/usr/bin/env bash\nexit 0\n")
    venv_py.chmod(0o755)
    env["LLAMA_LAUNCHER_VENV_PY"] = str(venv_py)
    return env


def _paths(tmp_path: Path):
    data = tmp_path / "data"
    desktop = data / "applications" / "llama-launcher.desktop"
    icon = data / "icons" / "hicolor" / "scalable" / "apps" / "llama-launcher.svg"
    return desktop, icon


@pytest.mark.skipif(not SCRIPT.exists(), reason="install script missing")
def test_install_creates_entry_and_icon(tmp_path):
    env = _env(tmp_path)
    r = subprocess.run(["bash", str(SCRIPT)], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    desktop, icon = _paths(tmp_path)
    assert icon.exists(), "icon was not installed"
    assert icon.read_bytes() == SRC_SVG.read_bytes(), "installed icon differs from asset"
    assert desktop.exists(), "desktop entry was not created"
    text = desktop.read_text()
    # Icon must point at the installed file by absolute path so a changed SVG
    # actually shows up regardless of icon-theme cache staleness.
    assert f"Icon={icon}" in text, f"Icon line not pointing at installed file:\n{text}"


@pytest.mark.skipif(not SCRIPT.exists(), reason="install script missing")
def test_uninstall_removes_entry_and_icon(tmp_path):
    env = _env(tmp_path)
    subprocess.run(["bash", str(SCRIPT)], env=env, check=True,
                   capture_output=True, text=True)
    subprocess.run(["bash", str(SCRIPT), "--uninstall"], env=env, check=True,
                   capture_output=True, text=True)
    desktop, icon = _paths(tmp_path)
    assert not desktop.exists()
    assert not icon.exists()


@pytest.mark.skipif(not SCRIPT.exists(), reason="install script missing")
def test_desktop_exec_quotes_python_path_with_spaces(tmp_path):
    """A venv path containing a space must stay one token in Exec= per the
    Desktop Entry spec, or the menu entry silently fails to launch."""
    env = dict(os.environ)
    env["HOME"] = str(tmp_path / "home")
    env["XDG_DATA_HOME"] = str(tmp_path / "data")
    env["PATH"] = f"{_stub_dir(tmp_path)}:{env['PATH']}"
    venv_py = tmp_path / "my venv" / "bin" / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("#!/usr/bin/env bash\nexit 0\n")
    venv_py.chmod(0o755)
    env["LLAMA_LAUNCHER_VENV_PY"] = str(venv_py)
    r = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    desktop, _ = _paths(tmp_path)
    text = desktop.read_text()
    # The spec quotes an executable path containing spaces with double quotes.
    assert f'Exec="{venv_py}" -m llama_launcher.app' in text, \
        f"Exec line not spec-quoted for a spaced path:\n{text}"


@pytest.mark.skipif(not SCRIPT.exists(), reason="install script missing")
def test_desktop_categories_has_one_main_category(tmp_path):
    """Two main categories (e.g. Development;Utility;) make the entry show up
    twice in spec-following menus (GNOME/XFCE/MATE)."""
    MAIN = {"AudioVideo", "Audio", "Video", "Development", "Education", "Game",
            "Graphics", "Network", "Office", "Science", "Settings", "System",
            "Utility"}
    r = subprocess.run(["bash", str(SCRIPT)], env=_env(tmp_path),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    desktop, _ = _paths(tmp_path)
    line = next(l for l in desktop.read_text().splitlines()
                if l.startswith("Categories="))
    cats = [c for c in line.split("=", 1)[1].split(";") if c]
    assert len(MAIN.intersection(cats)) == 1, f"{line} lists >1 main category"
