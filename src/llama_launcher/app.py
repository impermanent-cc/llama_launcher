import argparse
import json
import shlex
import sys

from llama_launcher.core.command_builder import build_command
from llama_launcher.core.report import redact_secrets
from llama_launcher.core.validation import dial_host, validate
from llama_launcher.services import headless
from llama_launcher.services import api_key as api_key_store
from llama_launcher.services.native import native_binary_ok_for
from llama_launcher.services.runtime import binary_available
from llama_launcher.services.terminal import DEFAULT_TEMPLATE, build_terminal_argv
from llama_launcher.store.profiles import (
    default_base_dir, list_profiles, load_config, resolve_member_pairs,
)


def _resolve_and_gate(action, profile_name, base_dir):
    """(profile, exit_code, message). profile is None exactly when exit_code is set."""
    profiles = {p.name: p for p in list_profiles(base_dir)}
    if not profiles:
        return None, 2, f"No saved profiles in {base_dir}."
    if profile_name is None:
        profile_name = load_config(base_dir).get("last_profile")
        if profile_name is None or profile_name not in profiles:
            return None, 2, f"Specify --profile NAME. Available: {', '.join(sorted(profiles))}"
    if profile_name not in profiles:
        return None, 2, f"Profile '{profile_name}' not found. Available: {', '.join(sorted(profiles))}"
    p = profiles[profile_name]
    # A launch will create the key if absent, so the exposure guard is satisfied
    # for launch; for stop/health, reflect whether a key actually exists today.
    api_key_present = (action == "launch") or bool(api_key_store.resolve_api_key(base_dir, p))
    members = resolve_member_pairs(p.members, base_dir)
    errs = [i for i in validate(p, binary_found=binary_available(p.runtime.binary),
                                members=members,
                                api_key_present=api_key_present,
                                native_binary_ok=native_binary_ok_for(p)) if i.level == "error"]
    if errs:
        return None, 2, "; ".join(i.message for i in errs)
    return p, None, None


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

    issues = validate(p, binary_found=binary_available(p.runtime.binary),
                     native_binary_ok=native_binary_ok_for(p))

    inner = build_command(p)
    template = load_config(base_dir).get("terminal", DEFAULT_TEMPLATE)
    term = build_terminal_argv(inner, template)

    # Redact the api-key: --dry-run output is exactly what users paste into bug
    # reports. Pass the known key value too, so a key without an `sk-` prefix or
    # with a space is caught as well.
    known = [str(p.settings.get("api-key", ""))]
    print("# Container command:")
    print(redact_secrets(shlex.join(inner), known=known))
    print("# Terminal invocation:")
    print(redact_secrets(shlex.join(term), known=known))

    if issues:
        print("# Validation:")
        for issue in issues:
            print(f"[{issue.level}] {issue.message}")

    return 1 if any(i.level == "error" for i in issues) else 0


def _emit(as_json, action, exit_code, *, status=None, name=None, host=None,
          port=None, warnings=(), error=None, text_out=None, text_err=None):
    if as_json:
        obj = {"action": action, "ok": exit_code == 0, "status": status,
               "name": name, "host": host, "port": port,
               "warnings": list(warnings), "error": error}
        print(json.dumps(obj))
    else:
        for w in warnings:
            print(w, file=sys.stderr)
        if text_err:
            print(text_err, file=sys.stderr)
        if text_out:
            print(text_out)
    return exit_code


def _do_launch(p, base_dir, wait, as_json=False):
    res = headless.launch(p, base_dir, p.runtime.binary)
    if not res.ok:
        return _emit(as_json, "launch", 1, name=res.name, host=res.host,
                     port=res.port, warnings=res.warnings, error=res.error,
                     text_err=f"'{p.name}' failed to start: {res.error}")
    if wait is None:
        return _emit(as_json, "launch", 0, status="started", name=res.name,
                     host=res.host, port=res.port, warnings=res.warnings,
                     text_out=f"'{p.name}' started ({res.name}) on {res.host}:{res.port}")
    if headless.wait_ready(dial_host(res.host), res.port, timeout=wait):
        return _emit(as_json, "launch", 0, status="ready", name=res.name,
                     host=res.host, port=res.port, warnings=res.warnings,
                     text_out=f"'{p.name}' ready on {res.host}:{res.port}")
    return _emit(as_json, "launch", 5, status="started", name=res.name,
                 host=res.host, port=res.port, warnings=res.warnings,
                 error=f"not ready after {int(wait)}s",
                 text_err=f"'{p.name}' started but not ready after {int(wait)}s")


_NATIVE_CLI_MSG = "native launch is GUI-only in this version"


def _do_stop(p, base_dir, as_json=False):
    name = headless._container_name(p)
    host = p.runtime.bind_host
    port = p.settings.get("port", 8080)
    if p.runtime.launch_mode == "native":
        return _emit(as_json, "stop", 1, name=name, host=host, port=port,
                     error=_NATIVE_CLI_MSG,
                     text_err=f"'{p.name}' is a native profile; {_NATIVE_CLI_MSG}")
    if headless.stop_router(p, p.runtime.binary, timeout=p.runtime.stop_timeout):
        return _emit(as_json, "stop", 0, status="stopped", name=name,
                     host=host, port=port,
                     text_out=f"'{p.name}' stopped")
    return _emit(as_json, "stop", 1, name=name, host=host, port=port,
                 error="failed to stop",
                 text_err=f"'{p.name}' failed to stop")


_HEALTH_EXIT = {"running": 0, "loading": 3}


def _do_health(p, base_dir, as_json=False):
    if p.runtime.launch_mode == "native":
        return _emit(as_json, "health", 1,
                     status="unknown", name=headless._container_name(p),
                     host=p.runtime.bind_host, port=p.settings.get("port", 8080),
                     error=_NATIVE_CLI_MSG,
                     text_err=f"'{p.name}' is a native profile; {_NATIVE_CLI_MSG}")
    status = headless.router_status(p, p.runtime.binary)
    display = "ready" if status == "running" else status
    return _emit(as_json, "health", _HEALTH_EXIT.get(status, 4),
                 status=display, name=headless._container_name(p),
                 host=p.runtime.bind_host, port=p.settings.get("port", 8080),
                 text_out=f"health: {display}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument("--profile", default=None)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--launch", action="store_true")
    group.add_argument("--stop", action="store_true")
    group.add_argument("--health", action="store_true")
    parser.add_argument("--wait", nargs="?", const=60.0, type=float, default=None)
    parser.add_argument("--json", action="store_true")

    args, unknown = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

    if args.dry_run:
        return dry_run(args.profile)

    if args.launch or args.stop or args.health:
        base = default_base_dir()
        action = "launch" if args.launch else "stop" if args.stop else "health"
        p, code, msg = _resolve_and_gate(action, args.profile, base)
        if code is not None:
            return _emit(args.json, action, code, name=args.profile,
                         error=msg, text_err=msg)
        if args.launch:
            return _do_launch(p, base, args.wait, args.json)
        if args.stop:
            return _do_stop(p, base, args.json)
        return _do_health(p, base, args.json)

    # GUI path — only import Qt here so that importing app.py never constructs QApplication.
    from PySide6.QtWidgets import QApplication
    from llama_launcher.ui.main_window import MainWindow
    from llama_launcher.ui.icon import app_icon

    app = QApplication([sys.argv[0]] + unknown)
    # Identity so the KDE/Wayland taskbar maps the window to the .desktop entry
    # (correct icon + pin-to-taskbar). Must match the installed .desktop basename.
    app.setApplicationName("Llama Launcher")
    app.setApplicationDisplayName("Llama Launcher")
    app.setDesktopFileName("llama-launcher")
    # Apply our own SVG to every window and the taskbar; the .desktop
    # association alone does not set the running window's icon, so without
    # this the window shows the desktop's default icon.
    app.setWindowIcon(app_icon())
    win = MainWindow()
    # Only keep the app alive after the last window closes when the user has
    # opted into minimize-to-tray; otherwise closing the window quits.
    app.setQuitOnLastWindowClosed(not win._minimize_to_tray)
    win.resize(1100, 760)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
