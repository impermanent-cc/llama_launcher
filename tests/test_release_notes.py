"""Integration tests for scripts/release-notes.sh.

These run the real script against fixture changelogs via the CHANGELOG override,
so they don't break every time the repo's own CHANGELOG.md gains an entry -- with
one exception at the bottom that deliberately checks the real file.
"""
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "release-notes.sh"

FIXTURE = """\
# Changelog

## [Unreleased]

### Added

### Fixed

## [0.2.0] - 2026-09-09

### Added
- A second thing.

### Changed

## [0.1.0] - 2026-09-01

First release, described in prose.

[0.1.0]: https://example.invalid/releases/tag/v0.1.0
"""


def _run(changelog: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(SCRIPT), *args],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "CHANGELOG": str(changelog)},
    )


@pytest.fixture
def changelog(tmp_path: Path) -> Path:
    p = tmp_path / "CHANGELOG.md"
    p.write_text(FIXTURE)
    return p


def test_extracts_only_the_requested_section(changelog: Path):
    r = _run(changelog, "0.1.0")
    assert r.returncode == 0, r.stderr
    assert r.stdout == "First release, described in prose.\n"


def test_stops_before_the_link_reference_block(changelog: Path):
    assert "example.invalid" not in _run(changelog, "0.1.0").stdout


def test_leading_v_is_optional(changelog: Path):
    assert _run(changelog, "v0.2.0").stdout == _run(changelog, "0.2.0").stdout


def test_drops_subheadings_that_were_never_filled_in(changelog: Path):
    out = _run(changelog, "0.2.0").stdout
    assert "### Added" in out and "- A second thing." in out
    assert "### Changed" not in out


def test_a_skeleton_only_section_is_not_publishable(changelog: Path):
    """An Unreleased section holding bare headings must not cut empty notes."""
    r = _run(changelog, "Unreleased")
    assert r.returncode != 0
    assert not r.stdout.strip()


def test_unknown_version_fails(changelog: Path):
    assert _run(changelog, "9.9.9").returncode != 0


def test_missing_version_argument_fails(changelog: Path):
    r = _run(changelog, )
    assert r.returncode != 0
    assert "usage:" in r.stderr


def test_missing_changelog_fails(tmp_path: Path):
    r = _run(tmp_path / "absent.md", "0.1.0")
    assert r.returncode != 0
    assert "no changelog" in r.stderr


def test_repo_changelog_has_notes_for_the_current_version():
    """The version in pyproject.toml must be releasable from the real changelog."""
    pyproject = (REPO / "pyproject.toml").read_text()
    version = next(
        line.split("=", 1)[1].strip().strip('"')
        for line in pyproject.splitlines()
        if line.startswith("version")
    )
    r = _run(REPO / "CHANGELOG.md", version)
    assert r.returncode == 0, f"CHANGELOG.md has no section for {version}"
    assert r.stdout.strip()
