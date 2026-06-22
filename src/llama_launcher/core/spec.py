import re
from dataclasses import dataclass, field


@dataclass
class Mount:
    host: str
    container: str
    role: str = "custom"          # "model" | "workspace" | "custom"
    mode: str = "ro"              # "ro" | "rw"
    selinux: str | None = None    # None | "z" | "Z"
    workdir: bool = False


@dataclass
class LoraRef:
    path: str
    scale: float = 1.0


@dataclass
class Runtime:
    binary: str = "podman"        # "podman" | "docker"
    gpu_mode: str = "cdi"         # "cdi" | "gpus-all" | "none"
    selinux_label_disable: bool = False
    extra_run_args: str = ""


@dataclass
class Profile:
    name: str
    image: str = ""
    runtime: Runtime = field(default_factory=Runtime)
    mounts: list[Mount] = field(default_factory=list)
    model: str = ""
    mmproj: str | None = None
    loras: list[LoraRef] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    raw_args: str = ""


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower())
    return s.strip("-")
