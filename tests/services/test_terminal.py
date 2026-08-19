import pytest

from llama_launcher.services import terminal
from llama_launcher.services.terminal import (
    build_terminal_argv, detect_terminal, DEFAULT_TEMPLATE, NoTerminalError,
)


# --- backward-compat: existing behavior unchanged ---

def test_build_terminal_argv_konsole():
    argv = build_terminal_argv(["podman", "run", "--rm", "img"])
    assert argv[:4] == ["konsole", "--hold", "-e", "bash"]
    assert argv[4] == "-lc"
    # last element is the single quoted command string
    assert argv[5] == "podman run --rm img"


def test_build_terminal_argv_quotes_spaces():
    argv = build_terminal_argv(["podman", "run", "-v", "/a b:/c"])
    assert "'/a b:/c'" in argv[-1]


def test_custom_template():
    argv = build_terminal_argv(["echo", "hi"], template="xterm -e bash -lc {cmd}")
    assert argv[0] == "xterm"
    assert argv[-1] == "echo hi"


# --- shell_hold: portable "keep window open" for terminals without native --hold ---

def test_shell_hold_appends_read_to_command_string():
    argv = build_terminal_argv(["podman", "run", "img"],
                               template="ptyxis -- bash -lc {cmd}", shell_hold=True)
    assert argv[:3] == ["ptyxis", "--", "bash"]
    cmd = argv[-1]
    assert cmd.startswith("podman run img")
    assert "read" in cmd  # holds the window open until Enter


def test_no_shell_hold_leaves_command_bare():
    argv = build_terminal_argv(["podman", "run", "img"],
                               template="ptyxis -- bash -lc {cmd}", shell_hold=False)
    assert argv[-1] == "podman run img"


# --- {bashcmd}: single-string -e terminals (kgx, tilix, xfce4) ---

def test_bashcmd_placeholder_is_single_token():
    argv = build_terminal_argv(["echo", "hi"], template="kgx -e {bashcmd}")
    assert argv[0] == "kgx" and argv[1] == "-e"
    # the whole "bash -lc '<cmd>'" is ONE argv token for a -e that takes a string
    assert len(argv) == 3
    assert argv[2] == "bash -lc 'echo hi'"


# --- detection ---

def _which_only(*present):
    names = set(present)
    return lambda b: (f"/usr/bin/{b}" if b in names else None)


def test_detect_prefers_konsole_when_present():
    tmpl, hold = detect_terminal(which=_which_only("konsole", "xterm", "ptyxis"))
    assert tmpl.startswith("konsole")
    assert hold is False  # konsole has native --hold


def test_detect_falls_back_to_ptyxis_on_gnome():
    tmpl, hold = detect_terminal(which=_which_only("ptyxis", "gnome-terminal", "xterm"))
    assert tmpl.startswith("ptyxis")
    assert hold is True  # ptyxis has no native hold -> portable shell hold


def test_detect_returns_none_when_no_terminal():
    assert detect_terminal(which=_which_only()) is None


def test_every_terminal_template_builds_a_valid_argv():
    # each registered terminal produces a non-empty argv whose first token is the binary
    for binary, template, _hold in terminal.TERMINALS:
        argv = build_terminal_argv(["echo", "x"], template=template)
        assert argv and argv[0] == binary


# --- launch: graceful failure instead of an unhandled FileNotFoundError ---

def test_launch_raises_no_terminal_when_none_available(monkeypatch):
    started = []
    monkeypatch.setattr(terminal.subprocess, "Popen",
                        lambda *a, **k: started.append(a))
    with pytest.raises(NoTerminalError):
        terminal.launch(["echo", "hi"], template=None, which=_which_only())
    assert started == []  # never tried to spawn


def test_launch_wraps_filenotfound_as_no_terminal(monkeypatch):
    def _boom(*a, **k):
        raise FileNotFoundError(2, "No such file or directory", "konsole")
    monkeypatch.setattr(terminal.subprocess, "Popen", _boom)
    with pytest.raises(NoTerminalError):
        terminal.launch(["echo", "hi"], template="konsole --hold -e bash -lc {cmd}")


def test_launch_uses_detected_terminal(monkeypatch):
    calls = {}
    monkeypatch.setattr(terminal.subprocess, "Popen",
                        lambda argv, **k: calls.setdefault("argv", argv))
    terminal.launch(["echo", "hi"], template=None, which=_which_only("ptyxis"))
    assert calls["argv"][0] == "ptyxis"
