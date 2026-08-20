import re
import shlex

from .settings_catalog import (
    CATALOG,
    IK_EXTRA_KV_CACHE_TYPES,
    ROUTER_ONLY_KEYS,
    router_catalog,
)
from collections.abc import Callable

from .spec import Profile, RpcWorker, slugify


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
                    detach: bool = False, connection: str = "",
                    network_host: bool = False) -> list[str]:
    rt = profile.runtime
    is_router = profile.mode == "router"

    argv = [rt.binary]
    if connection:
        argv += ["--connection", connection]
    argv += ["run"]
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

    if network_host:
        argv += ["--network", "host"]           # head shares host loopback for --rpc
    else:
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
    # An RPC pool head ALWAYS needs this override regardless of tag: a pool image
    # must carry both llama-server and ggml-rpc-server, so it is a full-style
    # (tools.sh-entrypoint) build, yet a user's custom tag (e.g. `…:rpc-cuda`)
    # won't match the full/light tag heuristic. Verified live 2026-08-20: without
    # this, the head ran tools.sh and printed usage instead of serving.
    has_entrypoint = any(a == "--entrypoint" or a.startswith("--entrypoint=") for a in extra)
    force_server = needs_server_entrypoint(profile.image) or profile.runtime.launch_mode == "rpc"
    if force_server and not has_entrypoint:
        argv += ["--entrypoint", _SERVER_ENTRYPOINT]

    argv.append(profile.image)
    return argv


def _render_setting(setting, value) -> list[str]:
    if setting.type == "bool":
        return [setting.flag] if value else []
    # An empty/blank value is meaningless as a flag argument -- emit nothing,
    # mirroring the False-bool case. This bites cleared string fields whose
    # default is non-empty (cors-origins defaults to "*", so clearing stores ""
    # -- distinct from the default -- and must not reach argv as `--cors-origins `
    # blank). It also guards the CLI/headless path, which skips the UI's
    # default-gating: a profile JSON carrying tools="" or an empty enum would
    # otherwise emit a dangling flag. No non-bool flag ever wants an empty arg;
    # numeric 0 / -1 stringify non-empty and are preserved.
    if not str(value).strip():
        return []
    return [setting.flag, str(value)]


def _owned_server_pairs(profile: Profile, catalog: dict, host: str = "0.0.0.0") -> list:
    pairs: list = []
    if profile.model:
        pairs.append(("-m", profile.model))
    if profile.mmproj:
        pairs.append(("--mmproj", profile.mmproj))
    for lora in profile.loras:
        if lora.scale is None or lora.scale == 1.0:
            pairs.append(("--lora", lora.path))
        else:
            pairs.append(("--lora-scaled", f"{lora.path}:{lora.scale}"))

    if profile.draft_model:
        pairs.append(("--spec-draft-model", profile.draft_model))

    port = profile.settings.get("port", 8080)
    # --load-mode supersedes the legacy --no-mmap/--mlock flags upstream; mixing
    # them makes llama.cpp warn and only honour the last. When load-mode is set
    # (it's only stored when non-default), drop the legacy flags so argv carries
    # one or the other, never both. Enforced here, not just in the UI, so the
    # CLI/headless path (which skips the form) stays consistent too.
    suppress = {"no-mmap", "mlock"} if "load-mode" in profile.settings else set()
    # Emit changed settings in catalog order, skipping port (handled below).
    for key, setting in catalog.items():
        if key == "port":
            continue
        if key in suppress:
            continue
        # Router-only flags are rejected by a single-model llama-server. The UI
        # filters them out by mode, but profile JSON written before that
        # filtering existed can still carry one.
        if key in ROUTER_ONLY_KEYS:
            continue
        # Engine-gated flags (ik_llama.cpp) must never reach a mainline launch.
        # current_profile() filters the UI path; this mirrors it for the
        # headless/CLI path, which feeds profile.settings straight from JSON.
        if setting.engine != "any" and setting.engine != profile.runtime.engine:
            continue
        if key in profile.settings:
            value = profile.settings[key]
            # ik-only KV-cache quant VALUES (q6_0/q8_KV) layer onto the shared,
            # engine="any" cache-type-k/-v settings, so the engine skip above
            # doesn't catch them. The UI only offers these when the ik engine
            # extends the enum; drop them here on a non-ik launch so a
            # JSON-leftover value can't emit an argument mainline rejects.
            if (key in ("cache-type-k", "cache-type-v")
                    and value in IK_EXTRA_KV_CACHE_TYPES
                    and profile.runtime.engine != "ik_llama.cpp"):
                continue
            # An enum value equal to its own default is a "leave it at the
            # engine's default" sentinel (auto/off/model default). Re-emitting
            # it is redundant at best and, for ik's --mla-use "auto", invalid
            # (ik wants an int 0-3). The UI drops these via is_set(); mirror that
            # on the headless path. Scoped to enums so numeric defaults that are
            # legitimate values (e.g. sleep-idle-seconds -1) still emit.
            if setting.type == "enum" and value == setting.default:
                continue
            rendered = _render_setting(setting, value)
            if not rendered:                      # bool that is False -> emits nothing
                continue
            pairs.append((rendered[0], rendered[1] if len(rendered) > 1 else None))

    pairs.append(("--host", host))
    pairs.append(("--port", str(port)))
    return pairs


_SERVER_PROTECTED = {"--host", "--port"}
_REPEATABLE = {"--lora", "--lora-scaled"}


def _server_args(profile: Profile, catalog: dict, host: str = "0.0.0.0") -> list[str]:
    owned = _owned_server_pairs(profile, catalog, host)
    argv, _warnings = _merge_raw_args(
        owned, _parse_raw_pairs(profile.raw_args), _SERVER_PROTECTED, _REPEATABLE)
    return argv


def _owned_router_pairs(profile: Profile) -> list:
    pairs: list = []
    port = profile.settings.get("port", 8080)
    for key, setting in router_catalog().items():
        if key == "port" or key == "api-key":
            continue
        if key in profile.settings:
            rendered = _render_setting(setting, profile.settings[key])
            if not rendered:
                continue
            pairs.append((rendered[0], rendered[1] if len(rendered) > 1 else None))
    pairs.append(("--models-preset", CONTAINER_PRESET_PATH))
    pairs.append(("--api-key-file", CONTAINER_KEY_PATH))
    pairs.append(("--host", "0.0.0.0"))
    pairs.append(("--port", str(port)))
    return pairs


_ROUTER_PROTECTED = {"--host", "--port", "--models-preset", "--api-key-file"}


def _router_server_args(profile: Profile) -> list[str]:
    """Server args for a router: no model, host-level settings only."""
    owned = _owned_router_pairs(profile)
    argv, _warnings = _merge_raw_args(
        owned, _parse_raw_pairs(profile.raw_args), _ROUTER_PROTECTED, _REPEATABLE)
    return argv


def raw_arg_warnings(profile: Profile, catalog: dict = CATALOG) -> list[str]:
    """Collisions between profile.raw_args and the flags the launcher emits.

    Same merge as build_command, returning only the warnings (empty when the
    raw_args don't collide with any launcher-owned, non-repeatable flag).
    """
    raw_pairs = _parse_raw_pairs(profile.raw_args)
    if profile.mode == "router":
        owned, protected = _owned_router_pairs(profile), _ROUTER_PROTECTED
    else:
        owned, protected = _owned_server_pairs(profile, catalog), _SERVER_PROTECTED
    _argv, warnings = _merge_raw_args(owned, raw_pairs, protected, _REPEATABLE)
    return warnings


def build_command(profile: Profile, catalog: dict = CATALOG,
                  router_host_dir: str = "", detach: bool = False,
                  connection: str = "", rpc_endpoints: str = "") -> list[str]:
    if profile.mode == "router":
        return _run_level_args(profile, router_host_dir, connection=connection) \
            + _router_server_args(profile)
    if profile.runtime.launch_mode == "native":
        return [profile.runtime.native_binary] + _server_args(
            profile, catalog, host=profile.runtime.bind_host)
    if profile.runtime.launch_mode == "rpc":
        # Head runs locally (no --connection) and host-networked so it can
        # reach every worker over 127.0.0.1. --host must stay the profile's
        # configured bind_host: --network host drops the -p bind_host:port:port
        # translation, so _server_args' 0.0.0.0 default would otherwise expose
        # the head API on every interface on the LAN.
        return _run_level_args(profile, detach=True, connection="", network_host=True) \
            + _server_args(profile, catalog, host=profile.runtime.bind_host) \
            + ["--rpc", rpc_endpoints]
    return _run_level_args(profile, detach=detach, connection=connection) \
        + _server_args(profile, catalog)


def build_rpc_endpoints(workers: list[RpcWorker],
                        resolve: Callable[[RpcWorker], int]) -> str:
    """`--rpc` value: comma-joined 127.0.0.1:<port> for each worker.

    `resolve(worker) -> int` gives the head-facing port (local worker's own port,
    or a remote worker's ssh local-forward port). Loopback throughout because the
    head runs with --network host and reaches every worker over the host loopback.
    """
    return ",".join(f"127.0.0.1:{resolve(w)}" for w in workers)


# Current llama.cpp builds the RPC server as `ggml-rpc-server` (upstream
# tools/rpc/CMakeLists.txt: `set(TARGET ggml-rpc-server)`); the older
# `rpc-server` name no longer exists. Verified live 2026-08-20 against a
# GGML_RPC=ON image.
_RPC_ENTRYPOINT = "/app/ggml-rpc-server"


def build_worker_command(profile, worker, index, connection="", wport=None):
    """Argv to run one rpc-server worker container. Publishes only to the worker
    host's loopback (never the LAN); the head reaches it directly (local) or over
    an ssh -L tunnel (remote)."""
    rt = profile.runtime
    port = wport if wport is not None else worker.port
    argv = [rt.binary]
    if connection:
        argv += ["--connection", connection]
    argv += ["run", "-d", "--name", f"llama-{slugify(profile.name)}-rpc{index}"]
    argv += ["--label", f"llama-launcher.profile={profile.name}"]
    argv += ["--label", "llama-launcher.mode=rpc-worker"]
    argv += ["--label", f"llama-launcher.pool={profile.name}"]
    if worker.device.upper().startswith("CUDA"):
        if rt.gpu_mode == "cdi":
            argv += ["--device", "nvidia.com/gpu=all"]
        elif rt.gpu_mode == "gpus-all":
            argv += ["--gpus", "all"]
    argv += ["-p", f"127.0.0.1:{port}:{port}"]
    argv += ["--entrypoint", _RPC_ENTRYPOINT, profile.image]
    argv += ["-H", "0.0.0.0", "-p", str(port), "-d", worker.device]
    # NB: current `ggml-rpc-server` has no per-worker memory-budget flag (its
    # only options are -t/-d/-H/-p/-c; verified live 2026-08-20). `worker.mem_mb`
    # is therefore a preflight-only pledge (feeds pooled_fit + the overcommit
    # warning) and is deliberately NOT passed to the worker — an unknown arg
    # makes ggml-rpc-server exit with "unknown argument".
    return argv
