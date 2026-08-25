import os
from pathlib import Path


def write_private(path: Path, text: str) -> None:
    """Write `text` to `path` owner-only (0600).

    Opens with the 0600 mode from the start so there is no world-readable
    window (write_text() + a later chmod would leave one), then chmod's to
    tighten a pre-existing wider file too. Mirrors store.profiles.save_profile.
    These config files (nodes, config) carry no secrets, but owner-only is the
    right default and keeps every launcher-written file consistent.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.chmod(path, 0o600)
