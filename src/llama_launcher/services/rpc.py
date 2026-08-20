"""Coordinated RPC pool lifecycle: launch each worker (tunneled over ssh when
remote), gate on readiness, then launch the local head with the resolved
`--rpc` endpoints. Mirrors services.native's synchronous, dependency-injected
style so the whole orchestrator is unit-testable without real subprocess/
socket I/O."""
import socket
import subprocess
import time
from dataclasses import dataclass

from llama_launcher.core.command_builder import (
    build_command, build_rpc_endpoints, build_worker_command,
)
from llama_launcher.core.nodes import (
    LOCAL_NODE, Node, connection_for, valid_ssh_target,
)
from llama_launcher.core.spec import slugify
from llama_launcher.services import runtime as rt_svc
from llama_launcher.store.nodes import get_node

_TUNNELS: dict = {}     # pool name -> list of Popen handles


def tunnel_argv(ssh_target, lport, wport) -> list:
    if not valid_ssh_target(ssh_target):
        raise ValueError(f"unsafe ssh target: {ssh_target!r}")
    return ["ssh", "-N", "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=15",
            "-L", f"127.0.0.1:{lport}:127.0.0.1:{wport}", ssh_target]


def alloc_local_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def wait_ready(port, connect, attempts=50, delay=0.1) -> bool:
    for _ in range(attempts):
        try:
            conn = connect(("127.0.0.1", port))
            try:
                conn.close()
            except Exception:
                pass
            return True
        except OSError:
            if delay:
                time.sleep(delay)
    return False


@dataclass
class PoolResult:
    ok: bool
    error: str = ""
    head_argv: list = None


def _default_connect(addr):
    return socket.create_connection(addr, timeout=1.0)


def launch_pool(profile, base_dir, *, run=None, popen=None, connect=None,
                alloc_port=None) -> PoolResult:
    run = run or (lambda argv: rt_svc._run(argv).returncode)
    popen = popen or (lambda argv: subprocess.Popen(argv))
    connect = connect or _default_connect
    alloc_port = alloc_port or alloc_local_port

    # A relaunch of the same pool without an intervening stop_pool must not
    # orphan the previous call's ssh tunnel handles.
    for t in _TUNNELS.get(profile.name, []):
        try:
            t.terminate()
        except Exception:
            pass

    workers = profile.runtime.rpc_workers
    tunnels = []
    resolved = {}                       # id(worker) -> head-facing port
    for i, w in enumerate(workers):
        node = get_node(base_dir, w.node) or LOCAL_NODE
        conn = connection_for(node)
        if run(build_worker_command(profile, w, i, connection=conn)) != 0:
            _teardown(profile, base_dir, workers[:i], tunnels, run)
            return PoolResult(False, f"worker {i} on '{w.node}' failed to start")
        if node.kind == "remote":
            lport = alloc_port()
            try:
                tunnel = popen(tunnel_argv(node.ssh_target, lport, w.port))
            except ValueError as exc:
                _teardown(profile, base_dir, workers[:i + 1], tunnels, run)
                return PoolResult(False, f"worker {i} on '{w.node}': {exc}")
            tunnels.append(tunnel)
            head_port = lport
        else:
            head_port = w.port
        resolved[id(w)] = head_port
        if not wait_ready(head_port, connect):
            _teardown(profile, base_dir, workers[:i + 1], tunnels, run)
            return PoolResult(False, f"worker {i} on '{w.node}' never became ready")
    _TUNNELS[profile.name] = tunnels
    endpoints = build_rpc_endpoints(workers, lambda w: resolved[id(w)])
    head = build_command(profile, rpc_endpoints=endpoints)
    if run(head) != 0:
        _teardown(profile, base_dir, workers, tunnels, run)
        _TUNNELS.pop(profile.name, None)
        return PoolResult(False, "head llama-server failed to start")
    return PoolResult(True, head_argv=head)


def _teardown(profile, base_dir, workers, tunnels, run):
    to = profile.runtime.stop_timeout
    for i, w in enumerate(workers):
        node = get_node(base_dir, w.node) or LOCAL_NODE
        run(rt_svc.stop_argv(f"llama-{slugify(profile.name)}-rpc{i}",
                             profile.runtime.binary, timeout=to,
                             connection=connection_for(node)))
    for t in tunnels:
        try:
            t.terminate()
        except Exception:
            pass


def stop_pool(profile, base_dir, *, run=None) -> None:
    run = run or (lambda argv: rt_svc._run(argv).returncode)
    to = profile.runtime.stop_timeout
    run(rt_svc.stop_argv(f"llama-{slugify(profile.name)}", profile.runtime.binary,
                         timeout=to, connection=""))
    for i, w in enumerate(profile.runtime.rpc_workers):
        node = get_node(base_dir, w.node) or LOCAL_NODE
        run(rt_svc.stop_argv(f"llama-{slugify(profile.name)}-rpc{i}",
                             profile.runtime.binary, timeout=to,
                             connection=connection_for(node)))
    for t in _TUNNELS.pop(profile.name, []):
        try:
            t.terminate()
        except Exception:
            pass
