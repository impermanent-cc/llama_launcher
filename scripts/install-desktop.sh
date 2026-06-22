#!/usr/bin/env bash
# Install a Llama Launcher entry into your desktop menu (KDE/GNOME/etc.) so you
# can pin it to the taskbar and stop typing the venv command.
#
#   ./scripts/install-desktop.sh           # install
#   ./scripts/install-desktop.sh --uninstall
#
# It points the launcher at THIS repo's .venv, so create the venv first:
#   python -m venv .venv && .venv/bin/pip install -e .
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$REPO/.venv/bin/python"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}"
DESKTOP_DIR="$DATA_DIR/applications"
ICON_DIR="$DATA_DIR/icons/hicolor/scalable/apps"
DESKTOP_FILE="$DESKTOP_DIR/llama-launcher.desktop"
ICON_FILE="$ICON_DIR/llama-launcher.svg"

if [[ "${1:-}" == "--uninstall" ]]; then
    rm -f "$DESKTOP_FILE" "$ICON_FILE"
    update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true
    echo "Removed Llama Launcher desktop entry."
    exit 0
fi

if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: $VENV_PY not found." >&2
    echo "Create the venv first:  python -m venv .venv && .venv/bin/pip install -e ." >&2
    exit 1
fi

mkdir -p "$DESKTOP_DIR" "$ICON_DIR"
install -m644 "$REPO/assets/llama-launcher.svg" "$ICON_FILE"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=Llama Launcher
GenericName=Local LLM launcher
Comment=Build and launch local llama.cpp servers in a terminal
Exec=$VENV_PY -m llama_launcher.app
Path=$REPO
Icon=llama-launcher
Terminal=false
Categories=Development;Utility;
StartupNotify=true
StartupWMClass=llama-launcher
EOF
chmod 644 "$DESKTOP_FILE"

update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true

echo "Installed:"
echo "  $DESKTOP_FILE"
echo "  $ICON_FILE"
echo
echo "Now search 'Llama Launcher' in your app menu, then right-click its icon"
echo "in the taskbar/launcher and choose 'Pin to Task Manager' (KDE)."
