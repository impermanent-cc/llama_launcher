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


def _run_level_args(profile: Profile) -> list[str]:
    rt = profile.runtime
    argv = [rt.binary, "run", "--rm", "--name", f"llama-{slugify(profile.name)}"]

    if rt.gpu_mode == "cdi":
        argv += ["--device", "nvidia.com/gpu=all"]
    elif rt.gpu_mode == "gpus-all":
        argv += ["--gpus", "all"]
    # "none" => nothing

    if rt.selinux_label_disable:
        argv.append("--security-opt=label=disable")

    port = profile.settings.get("port", 8080)
    argv += ["-p", f"127.0.0.1:{port}:{port}"]

    workdir = None
    for m in profile.mounts:
        opts = _mount_opts(m)
        spec = f"{m.host}:{m.container}:{opts}" if opts else f"{m.host}:{m.container}"
        argv += ["-v", spec]
        if m.workdir:
            workdir = m.container
    if workdir:
        argv += ["-w", workdir]

    if rt.extra_run_args.strip():
        argv += shlex.split(rt.extra_run_args)

    argv.append(profile.image)
    argv += ["-m", profile.model]
    return argv


def build_command(profile: Profile, catalog: dict = CATALOG) -> list[str]:
    return _run_level_args(profile)
