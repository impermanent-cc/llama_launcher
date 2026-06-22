import shutil
import subprocess


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def binary_available(binary: str) -> bool:
    return shutil.which(binary) is not None


def is_rootless(binary: str) -> bool:
    res = _run([binary, "info", "--format", "{{.Host.Security.Rootless}}"])
    return res.stdout.strip() == "true"


def container_state(name: str, binary: str) -> str:
    res = _run([binary, "inspect", "-f", "{{.State.Running}}", name])
    if res.returncode != 0:
        return "absent"
    return "running" if res.stdout.strip() == "true" else "stopped"


def stop(name: str, binary: str) -> None:
    _run([binary, "stop", name])
