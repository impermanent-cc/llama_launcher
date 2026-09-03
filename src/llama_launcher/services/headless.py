"""Qt-free router lifecycle for the headless CLI. Synchronous subprocess only."""
import subprocess
import time
from dataclasses import dataclass, field

from llama_launcher.core.command_builder import build_command
from llama_launcher.core.router_preset import render_preset
from llama_launcher.core.spec import profile_port, slugify
from llama_launcher.core.validation import dial_host
from llama_launcher.services import api_key as api_key_store
from llama_launcher.services.health import derive_status, probe_health
from llama_launcher.services.runtime import container_state, rm_argv, stop_argv
from llama_launcher.store.profiles import resolve_member_pairs


def _container_name(profile) -> str:
    return f"llama-{slugify(profile.name)}"


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    """Blocking subprocess; OSError (e.g. binary missing) \u2192 rc 127, never raises."""
    try:
        return subprocess.run(argv, capture_output=True, text=True, check=False)
    except OSError as exc:
        return subprocess.CompletedProcess(argv, 127, "", str(exc))


@dataclass
class LaunchResult:
    ok: bool
    name: str
    host: str
    port: int
    warnings: list = field(default_factory=list)
    error: str | None = None


def launch_router(profile, base_dir, binary) -> LaunchResult:
    """Prepare preset + key, then `podman run -d` the router. Synchronous."""
    name = _container_name(profile)
    host = profile.runtime.bind_host
    port = profile_port(profile)

    pairs = resolve_member_pairs(profile.members, base_dir)
    result = render_preset(pairs)
    api_key_store.prepare_launch_key(base_dir, profile)
    api_key_store.write_preset(base_dir, profile.name, result.text)
    router_host_dir = str(api_key_store.router_dir(base_dir, profile.name))

    argv = build_command(profile, router_host_dir=router_host_dir)

    # Router mode omits --rm, so a stopped same-name container would fail the run
    # with "name already in use". Remove it first (synchronously).
    if container_state(name, binary) == "stopped":
        _run(rm_argv(name, binary))

    proc = _run(argv)
    if proc.returncode != 0:
        return LaunchResult(False, name, host, port, result.warnings, proc.stderr.strip())
    return LaunchResult(True, name, host, port, result.warnings, None)


def launch_server(profile, base_dir, binary) -> LaunchResult:
    """`podman run -d` a single-model server. No router preset/key-file prep:
    a server's key is the `api-key` command setting, not a file. `base_dir` is
    accepted for signature parity with launch_router and is unused."""
    name = _container_name(profile)
    host = profile.runtime.bind_host
    port = profile_port(profile)

    # Force detached/persistent so --stop/--health can address it by name
    # (the GUI's server launch is foreground --rm and cannot be driven headless).
    argv = build_command(profile, detach=True)

    # Detached persistence means a leftover stopped container collides with
    # "name already in use"; remove it first (synchronously).
    if container_state(name, binary) == "stopped":
        _run(rm_argv(name, binary))

    proc = _run(argv)
    if proc.returncode != 0:
        return LaunchResult(False, name, host, port, [], proc.stderr.strip())
    return LaunchResult(True, name, host, port, [], None)


def launch(profile, base_dir, binary) -> LaunchResult:
    """Dispatch to the router or server launcher by profile.mode."""
    if profile.runtime.launch_mode == "native":
        return LaunchResult(False, _container_name(profile),
                            profile.runtime.bind_host,
                            profile_port(profile),
                            [], "native launch is GUI-only in this version")
    if profile.runtime.launch_mode == "rpc":
        return LaunchResult(False, _container_name(profile),
                            profile.runtime.bind_host,
                            profile_port(profile),
                            [], "RPC pool launch is GUI-only in this version")
    if profile.mode == "router":
        return launch_router(profile, base_dir, binary)
    return launch_server(profile, base_dir, binary)


def stop_router(profile, binary, timeout: int = 10) -> bool:
    """Stop the router container. True on success or if it's already absent."""
    name = _container_name(profile)
    if container_state(name, binary) == "absent":
        return True
    return _run(stop_argv(name, binary, timeout)).returncode == 0


def router_status(profile, binary) -> str:
    """Container state + /health \u2192 a display status (see health.derive_status)."""
    name = _container_name(profile)
    cstate = container_state(name, binary)
    if cstate != "running":
        return derive_status(cstate, "down")
    health = probe_health(profile_port(profile),
                          host=dial_host(profile.runtime.bind_host))
    return derive_status(cstate, health)


def wait_ready(host, port, timeout: float = 60.0, interval: float = 1.0) -> bool:
    """Poll /health until ready or timeout. True iff it became ready in time."""
    deadline = time.monotonic() + timeout
    while True:
        if probe_health(port, host=host) == "ready":
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)
