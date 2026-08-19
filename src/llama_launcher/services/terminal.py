"""Launch a container command in a graphical terminal emulator.

The launcher's foreground (non-detached) container launch opens a terminal so
the user can watch llama-server's output. Terminals are DE-specific, so instead
of hardcoding one we keep an ordered candidate list and pick the first that is
actually installed (``shutil.which``). konsole is first, so KDE behaves exactly
as before; GNOME/others fall through to ptyxis, gnome-terminal, etc. A user can
override everything with a ``terminal`` template in config.

Two placeholders are supported in a template:
  {cmd}     -> the shell-quoted command, as argv following an ``-e``/``--`` that
               takes a real argv (konsole -e, gnome-terminal --, ptyxis --, ...).
  {bashcmd} -> the whole ``bash -lc '<cmd>'`` collapsed into ONE token, for
               terminals whose ``-e`` wants a single command STRING (kgx, tilix).

Terminals without a native "hold" flag (keep the window open after the process
exits) get a portable shell hold appended to the command instead.
"""
import shlex
import shutil
import subprocess

# Keep the window open after the process exits, on any terminal (the portable
# analog of konsole's --hold / xterm's -hold).
_HOLD_SUFFIX = '; printf "\\n[process exited -- press Enter to close] "; read _'

# (binary, template, shell_hold). shell_hold=True appends _HOLD_SUFFIX to the
# command for terminals that have no native hold flag. Ordered by preference;
# konsole first so an existing KDE setup is byte-for-byte unchanged.
TERMINALS: list[tuple[str, str, bool]] = [
    ("konsole", "konsole --hold -e bash -lc {cmd}", False),
    ("ptyxis", "ptyxis -- bash -lc {cmd}", True),
    ("gnome-terminal", "gnome-terminal -- bash -lc {cmd}", True),
    ("kgx", "kgx -e {bashcmd}", True),
    ("xfce4-terminal", "xfce4-terminal --hold -x bash -lc {cmd}", False),
    ("xterm", "xterm -hold -e bash -lc {cmd}", False),
    ("alacritty", "alacritty -e bash -lc {cmd}", True),
    ("kitty", "kitty bash -lc {cmd}", True),
    ("tilix", "tilix -e {bashcmd}", True),
]

# Backwards-compatible default (konsole) for callers that pass no template.
DEFAULT_TEMPLATE = TERMINALS[0][1]


class NoTerminalError(RuntimeError):
    """No usable terminal emulator was found (or the chosen one failed to start)."""


def detect_terminal(which=shutil.which) -> tuple[str, bool] | None:
    """Return (template, shell_hold) for the first installed terminal, or None.

    `which` is injectable for testing.
    """
    for binary, template, shell_hold in TERMINALS:
        if which(binary) is not None:
            return template, shell_hold
    return None


def build_terminal_argv(command_argv: list[str], template: str = DEFAULT_TEMPLATE,
                        shell_hold: bool = False) -> list[str]:
    inner = shlex.join(command_argv)
    if shell_hold:
        inner += _HOLD_SUFFIX
    bashcmd = "bash -lc " + shlex.quote(inner)
    out: list[str] = []
    for token in shlex.split(template):
        if token == "{cmd}":
            out.append(inner)
        elif token == "{bashcmd}":
            out.append(bashcmd)
        else:
            out.append(token)
    return out


def launch(command_argv: list[str], template: str | None = None,
           shell_hold: bool = False, which=shutil.which) -> None:
    """Open command_argv in a terminal.

    template=None auto-detects an installed terminal (and its shell_hold); a
    non-None template is used verbatim (shell_hold as given -- a user-supplied
    template crafts its own hold). Raises NoTerminalError if no terminal is
    available or the chosen terminal binary cannot be started.
    """
    if template is None:
        detected = detect_terminal(which)
        if detected is None:
            raise NoTerminalError(
                "No terminal emulator found (tried: "
                + ", ".join(b for b, _t, _h in TERMINALS) + "). Install one, set a "
                "'terminal' command in the launcher config, or use 'Run detached' "
                "(or a native profile) -- neither needs a terminal.")
        template, shell_hold = detected
    argv = build_terminal_argv(command_argv, template, shell_hold)
    try:
        subprocess.Popen(argv, start_new_session=True)
    except FileNotFoundError as exc:
        raise NoTerminalError(
            f"Terminal '{argv[0]}' could not be started: {exc}. Set a 'terminal' "
            "command in the launcher config, or use 'Run detached'.") from exc
