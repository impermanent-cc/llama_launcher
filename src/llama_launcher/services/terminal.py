import shlex
import subprocess

DEFAULT_TEMPLATE = "konsole --hold -e bash -lc {cmd}"


def build_terminal_argv(command_argv: list[str], template: str = DEFAULT_TEMPLATE) -> list[str]:
    cmd_str = shlex.join(command_argv)
    out: list[str] = []
    for token in shlex.split(template):
        if token == "{cmd}":
            out.append(cmd_str)
        else:
            out.append(token)
    return out


def launch(command_argv: list[str], template: str = DEFAULT_TEMPLATE) -> None:
    argv = build_terminal_argv(command_argv, template)
    subprocess.Popen(argv, start_new_session=True)
