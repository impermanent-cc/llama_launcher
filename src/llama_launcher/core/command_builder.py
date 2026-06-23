import shlex

from .settings_catalog import CATALOG
from .spec import Profile, slugify


def _mount_opts(mount) -> str:
    parts = []
    if mount.mode:
        parts.append(mount.mode)
    if mount.selinux:
        parts.append(mount.selinux)
    return ",".join(parts)


# llama.cpp image tags whose entrypoint is the multi-tool dispatcher
# (/app/tools.sh), which rejects llama-server flags like -m. For these we point
# podman/docker straight at the llama-server binary instead.
_TOOLS_VARIANTS = ("full", "light")
_SERVER_ENTRYPOINT = "/app/llama-server"


def image_tag(image: str) -> str:
    """Return the tag portion of an image ref, or '' when none is present."""
    _, sep, tag = image.rpartition(":")
    if not sep or "/" in tag:        # ":" belonged to a host:port, or no tag
        return ""
    return tag


def needs_server_entrypoint(image: str) -> bool:
    """True for `full`/`light` (tool-dispatcher) image tags whose entrypoint
    must be overridden to the llama-server binary."""
    return image_tag(image).split("-", 1)[0] in _TOOLS_VARIANTS


def _run_level_args(profile: Profile) -> list[str]:
    rt = profile.runtime
    argv = [rt.binary, "run", "--rm", "--name", f"llama-{slugify(profile.name)}"]

    if rt.gpu_mode == "cdi":
        argv += ["--device", "nvidia.com/gpu=all"]
    elif rt.gpu_mode == "gpus-all":
        argv += ["--gpus", "all"]

    if rt.selinux_label_disable:
        argv.append("--security-opt=label=disable")

    port = profile.settings.get("port", 8080)
    argv += ["-p", f"127.0.0.1:{port}:{port}"]

    workdir = None
    for m in profile.mounts:
        if not m.host or not m.container:
            continue
        opts = _mount_opts(m)
        spec = f"{m.host}:{m.container}:{opts}" if opts else f"{m.host}:{m.container}"
        argv += ["-v", spec]
        if m.workdir:
            workdir = m.container
    if workdir:
        argv += ["-w", workdir]
        # The official llama.cpp images resolve their bundled shared libraries
        # relative to the default working dir (/app). Setting -w to a custom
        # workspace moves the CWD off /app and breaks the dynamic linker
        # ("libllama-server-impl.so: cannot open shared object file"), so pin
        # the library path explicitly whenever we change the working directory.
        argv += ["-e", "LD_LIBRARY_PATH=/app"]

    extra = shlex.split(rt.extra_run_args) if rt.extra_run_args.strip() else []
    argv += extra

    # Full/light images use the /app/tools.sh dispatcher (which rejects -m); point
    # them straight at llama-server unless the user already set an --entrypoint.
    has_entrypoint = any(a == "--entrypoint" or a.startswith("--entrypoint=") for a in extra)
    if needs_server_entrypoint(profile.image) and not has_entrypoint:
        argv += ["--entrypoint", _SERVER_ENTRYPOINT]

    argv.append(profile.image)
    return argv


def _render_setting(setting, value) -> list[str]:
    if setting.type == "bool":
        return [setting.flag] if value else []
    return [setting.flag, str(value)]


def _server_args(profile: Profile, catalog: dict) -> list[str]:
    argv: list[str] = []
    if profile.model:
        argv += ["-m", profile.model]
    if profile.mmproj:
        argv += ["--mmproj", profile.mmproj]
    for lora in profile.loras:
        if lora.scale is None or lora.scale == 1.0:
            argv += ["--lora", lora.path]
        else:
            argv += ["--lora-scaled", f"{lora.path}:{lora.scale}"]

    if profile.draft_model:
        argv += ["--spec-draft-model", profile.draft_model]

    port = profile.settings.get("port", 8080)
    # Emit changed settings in catalog order, skipping port (handled below).
    for key, setting in catalog.items():
        if key == "port":
            continue
        if key in profile.settings:
            argv += _render_setting(setting, profile.settings[key])

    argv += ["--host", "0.0.0.0", "--port", str(port)]

    if profile.raw_args.strip():
        argv += shlex.split(profile.raw_args)
    return argv


def build_command(profile: Profile, catalog: dict = CATALOG) -> list[str]:
    return _run_level_args(profile) + _server_args(profile, catalog)
