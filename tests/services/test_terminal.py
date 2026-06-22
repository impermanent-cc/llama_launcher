from llama_launcher.services.terminal import build_terminal_argv, DEFAULT_TEMPLATE


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
