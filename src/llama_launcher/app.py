import argparse
import shlex
import sys

from llama_launcher.core.command_builder import build_command
from llama_launcher.core.validation import validate
from llama_launcher.services.runtime import binary_available
from llama_launcher.services.terminal import DEFAULT_TEMPLATE, build_terminal_argv
from llama_launcher.store.profiles import default_base_dir, list_profiles, load_config


def dry_run(profile_name: str | None = None, base_dir=None) -> int:
    base_dir = base_dir or default_base_dir()

    profiles = {p.name: p for p in list_profiles(base_dir)}

    if not profiles:
        print(f"No saved profiles in {base_dir}.")
        return 2

    if profile_name is None:
        last = load_config(base_dir).get("last_profile")
        if last is None or last not in profiles:
            names = ", ".join(sorted(profiles))
            print(f"Specify --profile NAME. Available: {names}")
            return 2
        profile_name = last

    if profile_name not in profiles:
        names = ", ".join(sorted(profiles))
        print(f"Profile '{profile_name}' not found. Available: {names}")
        return 2

    p = profiles[profile_name]

    issues = validate(p, binary_found=binary_available(p.runtime.binary))

    inner = build_command(p)
    template = load_config(base_dir).get("terminal", DEFAULT_TEMPLATE)
    term = build_terminal_argv(inner, template)

    print("# Container command:")
    print(shlex.join(inner))
    print("# Terminal invocation:")
    print(shlex.join(term))

    if issues:
        print("# Validation:")
        for issue in issues:
            print(f"[{issue.level}] {issue.message}")

    return 1 if any(i.level == "error" for i in issues) else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument("--profile", default=None)

    args, unknown = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

    if args.dry_run:
        return dry_run(args.profile)

    # GUI path — only import Qt here so that importing app.py never constructs QApplication.
    from PySide6.QtWidgets import QApplication
    from llama_launcher.ui.main_window import MainWindow

    app = QApplication([sys.argv[0]] + unknown)
    # Identity so the KDE/Wayland taskbar maps the window to the .desktop entry
    # (correct icon + pin-to-taskbar). Must match the installed .desktop basename.
    app.setApplicationName("Llama Launcher")
    app.setApplicationDisplayName("Llama Launcher")
    app.setDesktopFileName("llama-launcher")
    win = MainWindow()
    # Only keep the app alive after the last window closes when the user has
    # opted into minimize-to-tray; otherwise closing the window quits.
    app.setQuitOnLastWindowClosed(not win._minimize_to_tray)
    win.resize(1100, 760)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
