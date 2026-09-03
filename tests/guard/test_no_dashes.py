"""Tracked text files carry no em dash, en dash, or doubled-hyphen prose.

The doubled-hyphen check reads Markdown files and comment lines only, with
backtick spans removed first, so a CLI separator in a command or a code
line never counts. CHECK_DOUBLE_HYPHEN turns it off for a repository whose
legacy prose separators are still being cleaned up.
"""

import re
from pathlib import Path

import pytest

EM_DASH = "\u2014"
EN_DASH = "\u2013"
CHECK_DOUBLE_HYPHEN = False
DOUBLE_HYPHEN_PROSE = re.compile(r"\w\s+--\s+\w|\w--\w")
BACKTICK_SPAN = re.compile(r"`[^`]*`")
COMMENT_LINE = re.compile(r"^\s*(#|//|\*|<!--)")
PROSE_SUFFIXES = {".md", ".rst", ".txt"}


def _text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def _prose_lines(path: Path, text: str):
    prose_file = path.suffix.lower() in PROSE_SUFFIXES
    for line_no, line in enumerate(text.splitlines(), start=1):
        if prose_file or COMMENT_LINE.match(line):
            yield line_no, BACKTICK_SPAN.sub("", line)


def test_no_em_or_en_dashes(tracked_files: list[Path], repo_root: Path) -> None:
    offenses = []
    for path in tracked_files:
        text = _text(path)
        if text is None:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if EM_DASH in line or EN_DASH in line:
                offenses.append(f"{path.relative_to(repo_root)}:{line_no}")
    assert not offenses, (
        "em or en dash found; use a comma, colon or hyphen:\n" + "\n".join(offenses)
    )


def test_no_double_hyphen_prose(tracked_files: list[Path], repo_root: Path) -> None:
    if not CHECK_DOUBLE_HYPHEN:
        pytest.skip("double-hyphen check is off for this repository")
    offenses = []
    for path in tracked_files:
        text = _text(path)
        if text is None:
            continue
        for line_no, line in _prose_lines(path, text):
            if DOUBLE_HYPHEN_PROSE.search(line):
                offenses.append(
                    f"{path.relative_to(repo_root)}:{line_no}: {line.strip()[:80]}"
                )
    assert not offenses, "doubled hyphen used as punctuation:\n" + "\n".join(offenses)
