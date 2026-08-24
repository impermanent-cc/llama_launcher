"""Application icon resolution.

The window/taskbar/tray icon is the app's own SVG (``assets/llama-launcher.svg``).
The app must call ``setWindowIcon(app_icon())`` explicitly; registering the
``.desktop`` entry only fixes the launcher-menu icon, not the running window's
icon, so without this the window falls back to the desktop's default icon.
"""

from pathlib import Path

from PySide6.QtGui import QIcon

_ICON_NAME = "llama-launcher.svg"
# Matches the basename installed into hicolor by scripts/install-desktop.sh,
# used as the QIcon.fromTheme() fallback below.
_THEME_NAME = "llama-launcher"


def _find_repo_icon() -> Path | None:
    """Locate ``assets/llama-launcher.svg`` by walking up from this file.

    Works for the editable/src-layout install the app ships as: the package
    lives at ``<repo>/src/llama_launcher/ui/`` and the asset at
    ``<repo>/assets/``.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "assets" / _ICON_NAME
        if candidate.is_file():
            return candidate
    return None


def app_icon() -> QIcon:
    """Return the application icon.

    Prefers the bundled SVG file; falls back to the installed theme icon (set
    up by install-desktop.sh) when the repo asset can't be found, e.g. a
    non-editable install. May be null if neither is available; callers that
    need a guaranteed-visible icon should provide their own fallback.
    """
    path = _find_repo_icon()
    if path is not None:
        icon = QIcon(str(path))
        if not icon.isNull():
            return icon
    return QIcon.fromTheme(_THEME_NAME)
