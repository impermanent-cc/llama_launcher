import re
from dataclasses import dataclass, field

# Seconds a container gets to shut down (SIGTERM) before it is force-killed
# (SIGKILL) — podman's own `stop -t` default. Single source for every layer's
# stop-grace default so they can't drift apart.
DEFAULT_STOP_TIMEOUT = 10


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
class RouterMember:
    """One model served by a router profile.

    `profile` names a saved profile whose settings become this model's preset
    section. `model_id` is what a harness puts in the request's "model" field;
    empty means "derive it from the profile name".
    """
    profile: str
    model_id: str = ""
    load_on_startup: bool = False
    # Kill-delay after an unload is requested: seconds llama.cpp waits before
    # forcing this model's process down. NOT an idle timer — idle unloading is
    # the server-wide `--sleep-idle-seconds` setting, a separate concept.
    stop_timeout: int = DEFAULT_STOP_TIMEOUT


@dataclass
class Runtime:
    binary: str = "podman"        # "podman" | "docker"
    gpu_mode: str = "cdi"         # "cdi" | "gpus-all" | "none"
    selinux_label_disable: bool = False
    extra_run_args: str = ""
    bind_host: str = "127.0.0.1"  # publish address; non-loopback exposes the port
    detached: bool = False        # GUI server launch: no terminal, Monitor-driven
    router_key_mode: str = "global"  # "global" (shared key) | "own" (per-profile key)
    engine: str = "llama.cpp"     # "llama.cpp" | "ik_llama.cpp"
    stop_timeout: int = DEFAULT_STOP_TIMEOUT  # `podman stop -t` grace before SIGKILL
    launch_mode: str = "container"  # "container" (podman/docker) | "native" (subprocess)
    native_binary: str = ""         # abs path to a prebuilt llama-server (native mode)
    node: str = "local"             # which registered node this profile launches on


@dataclass
class Profile:
    name: str
    image: str = ""
    runtime: Runtime = field(default_factory=Runtime)
    mounts: list[Mount] = field(default_factory=list)
    model: str = ""
    mmproj: str | None = None
    draft_model: str | None = None
    loras: list[LoraRef] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    raw_args: str = ""
    mode: str = "server"                                # "server" | "router"
    members: list[RouterMember] = field(default_factory=list)


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower())
    return s.strip("-")


def member_model_id(m: RouterMember) -> str:
    """The id a harness sends in its "model" field for this member."""
    return m.model_id or slugify(m.profile)
