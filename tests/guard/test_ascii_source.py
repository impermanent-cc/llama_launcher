"""Tracked text files are ASCII except the allowlisted data files.

Incidental non-ASCII is reworded; a functional glyph is spelled as an
escape such as \\u2212; a genuine data file is allowlisted in conftest.py.
"""

from pathlib import Path


def test_tracked_text_is_ascii(tracked_files: list[Path], repo_root: Path) -> None:
    offenses = []
    for path in tracked_files:
        data = path.read_bytes()
        if data.isascii():
            continue
        rel = path.relative_to(repo_root).as_posix()
        for line_no, line in enumerate(data.splitlines(), start=1):
            if not line.isascii():
                chars = sorted({f"U+{ord(c):04X}" for c in line.decode("utf-8", "replace") if ord(c) > 0x7F})
                offenses.append(f"{rel}:{line_no}: {' '.join(chars)}")
    assert not offenses, (
        "non-ASCII outside the allowlist; reword to ASCII, spell glyphs as "
        "escapes, or allowlist a genuine data file in conftest.py:\n" + "\n".join(offenses)
    )


def test_allowlist_entries_exist(repo_root: Path, allowlist: tuple[str, ...]) -> None:
    missing = [e for e in allowlist if not (repo_root / e).exists()]
    assert not missing, f"stale ALLOWLIST entries: {missing}"


def test_allowlist_entries_still_needed(
    repo_root: Path, allowlist: tuple[str, ...], binary_suffixes: set[str]
) -> None:
    stale = []
    for entry in allowlist:
        target = repo_root / entry
        paths = [p for p in target.rglob("*") if p.is_file()] if target.is_dir() else [target]
        texts = [p.read_bytes() for p in paths if p.suffix.lower() not in binary_suffixes]
        if not any(not t.isascii() or b"--" in t for t in texts):
            stale.append(entry)
    assert not stale, f"ALLOWLIST entries that no longer need it: {stale}"
