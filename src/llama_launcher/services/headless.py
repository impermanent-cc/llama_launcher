"""Qt-free router lifecycle for the headless CLI. Synchronous subprocess only."""
import subprocess
import time
from dataclasses import dataclass, field

from llama_launcher.core.command_builder import build_command
from llama_launcher.core.router_preset import render_preset
from llama_launcher.core.spec import slugify
from llama_launcher.core.validation import dial_host
from llama_launcher.services import api_key as api_key_store
from llama_launcher.services.health import derive_status, probe_health
from llama_launcher.services.runtime import container_state, rm_argv, stop_argv
from llama_launcher.store.profiles import resolve_member_pairs


def _container_name(profile) -> str:
    return f"llama-{slugify(profile.name)}"


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    """Blocking subprocess; OSError (e.g. binary missing) → rc 127, never raises."""
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
    port = profile.settings.get("port", 8080)

    pairs = resolve_member_pairs(profile.members, base_dir)
    result = render_preset(pairs)
    api_key_store.ensure_api_key(base_dir, profile.name)
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
