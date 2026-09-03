import re
from dataclasses import dataclass, field

# Seconds a container gets to shut down (SIGTERM) before it is force-killed
# (SIGKILL), podman's own `stop -t` default. Single source for every layer's
# stop-grace default so they can't drift apart.
DEFAULT_STOP_TIMEOUT = 10

# TCP port a profile serves on when it carries no explicit "port" setting.
# Matches llama-server's own default. Upstream has announced a future move to
# 9931 (ggml-org/llama.cpp#26508) but has NOT made it yet; llama_launcher always
# passes --port explicitly, so the flip cannot change how a launch behaves. When
# it lands, changing it here is the whole edit.
DEFAULT_PORT = 8080


@dataclass
class Mount:
    host: str
    container: str
    role: str = "custom"  # "model" | "workspace" | "custom"
    mode: str = "ro"  # "ro" | "rw"
    selinux: str | None = None  # None | "z" | "Z"
    workdir: bool = False


@dataclass
class LoraRef:
    path: str
    scale: float = 1.0


@dataclass(frozen=True)
class RpcWorker:
    node: str
    device: str = "CPU"  # "CPU" | "CUDA0" | "CUDA1" | ...
    mem_mb: int = 0  # --mem budget (0 = let rpc-server decide)
    port: int = 50052  # rpc-server port on the worker host


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
    # forcing this model's process down. NOT an idle timer; idle unloading is
    # the server-wide `--sleep-idle-seconds` setting, a separate concept.
    stop_timeout: int = DEFAULT_STOP_TIMEOUT


@dataclass
class Runtime:
    binary: str = "podman"  # "podman" | "docker"
    gpu_mode: str = "cdi"  # "cdi" | "gpus-all" | "none"
    selinux_label_disable: bool = False
    extra_run_args: str = ""
    bind_host: str = "127.0.0.1"  # publish address; non-loopback exposes the port
    detached: bool = False  # GUI server launch: no terminal, Monitor-driven
    router_key_mode: str = "global"  # "global" (shared key) | "own" (per-profile key)
    engine: str = "llama.cpp"  # "llama.cpp" | "ik_llama.cpp"
    stop_timeout: int = DEFAULT_STOP_TIMEOUT  # `podman stop -t` grace before SIGKILL
    launch_mode: str = (
        "container"  # "container" (podman/docker) | "native" (subprocess) | "rpc"
    )
    native_binary: str = ""  # abs path to a prebuilt llama-server (native mode)
    node: str = "local"  # which registered node this profile launches on
    rpc_workers: list[RpcWorker] = field(default_factory=list)  # RPC mode only


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
    mode: str = "server"  # "server" | "router"
    members: list[RouterMember] = field(default_factory=list)


def profile_port(profile: "Profile") -> int:
    """The port `profile` serves on: its explicit setting, else DEFAULT_PORT.

    Every layer (launch, monitor, benchmark, report, headless) reads the port
    through here so the default cannot drift between them.
    """
    return profile.settings.get("port", DEFAULT_PORT)


def slugify(name: str, fallback: str = "unnamed") -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    # An all-symbol / empty name would collapse to "" -> a ".json" filename or a
    # "llama-" container name; fall back to a stable placeholder.
    return s or fallback


def member_model_id(m: RouterMember) -> str:
    """The id a harness sends in its "model" field for this member."""
    return m.model_id or slugify(m.profile)
