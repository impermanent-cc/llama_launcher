import re
import shlex

from .settings_catalog import CATALOG, ROUTER_ONLY_KEYS, router_catalog
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

# Where the launcher-managed router files land inside the container.
CONTAINER_ROUTER_DIR = "/router"
CONTAINER_PRESET_PATH = f"{CONTAINER_ROUTER_DIR}/models.ini"
CONTAINER_KEY_PATH = f"{CONTAINER_ROUTER_DIR}/api-key"


# Structural launcher flags that are not catalog settings but that raw_args
# might collide with. Maps each spelling to its canonical (long) form.
_STRUCTURAL_ALIASES = {
    "-m": "--model",
    "--model": "--model",
    "--mmproj": "--mmproj",
    "--spec-draft-model": "--spec-draft-model",
    "--host": "--host",
    "--port": "--port",
    "--models-preset": "--models-preset",
    "--api-key-file": "--api-key-file",
}


def _build_alias_fold(catalog: dict) -> dict:
    """alias/spelling -> canonical long flag, from catalog + structural flags."""
    fold: dict = dict(_STRUCTURAL_ALIASES)
    for setting in catalog.values():
        fold[setting.flag] = setting.flag          # long form is its own canonical
        for alias in setting.aliases:
            fold[alias] = setting.flag
    return fold


_ALIAS_FOLD = _build_alias_fold(CATALOG)


def _canonical_flag(flag: str) -> str:
    return _ALIAS_FOLD.get(flag, flag)


# Same flag rule as router_preset._FLAG_RE: one/two dashes then a letter, so
# llama.cpp negative sentinels (-ngl -1, --top-n-sigma -1.5, --seed -1) are read
# as values, not flags.
_FLAG_RE = re.compile(r"^--?[A-Za-z]")


def _parse_raw_pairs(raw: str) -> list:
    """['--ctx-size', '8192', '--mlock'] -> [('--ctx-size','8192'), ('--mlock',None)].

    Order- and repeat-preserving (unlike router_preset.convert_raw_args, which
    collapses to a unique-key INI dict).
    """
    pairs: list = []
    if not raw.strip():
        return pairs
    tokens = shlex.split(raw)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not _FLAG_RE.match(tok):
            i += 1                      # stray positional; nothing to pair it with
            continue
        if "=" in tok:
            flag, _, value = tok.partition("=")
            pairs.append((flag, value))
            i += 1
            continue
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if nxt is not None and not _FLAG_RE.match(nxt):
            pairs.append((tok, nxt))
            i += 2
        else:
            pairs.append((tok, None))
            i += 1
    return pairs


def _flatten_pairs(pairs) -> list:
    argv: list = []
    for flag, value in pairs:
        argv.append(flag)
        if value is not None:
            argv.append(value)
    return argv


def _merge_raw_args(owned, raw_pairs, protected_canon, repeatable_canon):
    """Fold raw_args pairs onto the launcher's owned pairs.

    raw wins for owned single-valued flags (replaced in place); protected
    canon flags keep the launcher value; repeatable/unknown flags append.
    Returns (argv, warnings).
    """
    owned = [list(pair) for pair in owned]          # mutable copy for in-place override
    owned_index: dict = {}
    for pos, (flag, _value) in enumerate(owned):
        owned_index.setdefault(_canonical_flag(flag), pos)   # first owner wins the slot
    warnings: list = []
    extras: list = []
    for flag, value in raw_pairs:
        canon = _canonical_flag(flag)
        if canon in protected_canon:
            warnings.append(f"raw arg '{flag}' ignored; the launcher controls '{canon}'")
            continue
        if canon in owned_index and canon not in repeatable_canon:
            pos = owned_index[canon]
            old = owned[pos][1]
            owned[pos][1] = value
            if old is None:
                warnings.append(f"raw arg '{flag}' duplicates '{canon}'")
            else:
                shown = f"{flag} {value}" if value is not None else flag
                warnings.append(f"raw arg '{shown}' overrides '{canon}' (was {old})")
            continue
        extras.append((flag, value))
    return _flatten_pairs(owned) + _flatten_pairs(extras), warnings


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


def _run_level_args(profile: Profile, router_host_dir: str = "",
                    detach: bool = False) -> list[str]:
    rt = profile.runtime
    is_router = profile.mode == "router"

    argv = [rt.binary, "run"]
    # A router is a persistent headless host: run it detached, and keep the
    # container after exit so a crash leaves a readable exit code and logs.
    # The headless CLI passes detach=True to give a single-model server the
    # same persistence (so --stop/--health can find it by name); the GUI leaves
    # detach False, so a GUI server keeps its foreground --rm behavior.
    argv += ["-d"] if (is_router or detach) else ["--rm"]
    argv += ["--name", f"llama-{slugify(profile.name)}"]

    # Labels let the launcher find and reattach to its own containers on restart.
    argv += ["--label", f"llama-launcher.profile={profile.name}"]
    argv += ["--label", f"llama-launcher.mode={profile.mode}"]

    if rt.gpu_mode == "cdi":
        argv += ["--device", "nvidia.com/gpu=all"]
    elif rt.gpu_mode == "gpus-all":
        argv += ["--gpus", "all"]

    if rt.selinux_label_disable:
        argv.append("--security-opt=label=disable")

    port = profile.settings.get("port", 8080)
    argv += ["-p", f"{rt.bind_host}:{port}:{port}"]

    workdir = None
    for m in profile.mounts:
        if not m.host or not m.container:
            continue
        opts = _mount_opts(m)
        spec = f"{m.host}:{m.container}:{opts}" if opts else f"{m.host}:{m.container}"
        argv += ["-v", spec]
        if m.workdir:
            workdir = m.container

    if is_router and router_host_dir:
        # Honour the profile's SELinux preference like every other mount: on an
        # enforcing host an unlabelled mount gives the container EACCES on the
        # preset and key, and the router fails to start with a confusing error.
        selinux = next((m.selinux for m in profile.mounts if m.selinux), None)
        opts = f"ro,{selinux}" if selinux else "ro"
        argv += ["-v", f"{router_host_dir}:{CONTAINER_ROUTER_DIR}:{opts}"]

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
        # Router-only flags are rejected by a single-model llama-server. The UI
        # filters them out by mode, but profile JSON written before that
        # filtering existed can still carry one.
        if key in ROUTER_ONLY_KEYS:
            continue
        if key in profile.settings:
            argv += _render_setting(setting, profile.settings[key])

    argv += ["--host", "0.0.0.0", "--port", str(port)]

    if profile.raw_args.strip():
        argv += shlex.split(profile.raw_args)
    return argv


def _router_server_args(profile: Profile) -> list[str]:
    """Server args for a router: no model, host-level settings only."""
    argv: list[str] = []
    port = profile.settings.get("port", 8080)

    for key, setting in router_catalog().items():
        if key == "port" or key == "api-key":
            continue          # port is emitted below; the key comes from a file
        if key in profile.settings:
            argv += _render_setting(setting, profile.settings[key])

    argv += ["--models-preset", CONTAINER_PRESET_PATH]
    argv += ["--api-key-file", CONTAINER_KEY_PATH]
    argv += ["--host", "0.0.0.0", "--port", str(port)]

    if profile.raw_args.strip():
        argv += shlex.split(profile.raw_args)
    return argv


def build_command(profile: Profile, catalog: dict = CATALOG,
                  router_host_dir: str = "", detach: bool = False) -> list[str]:
    if profile.mode == "router":
        return _run_level_args(profile, router_host_dir) + _router_server_args(profile)
    return _run_level_args(profile, detach=detach) + _server_args(profile, catalog)
