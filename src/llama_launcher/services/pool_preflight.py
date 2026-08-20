"""RPC-pool preflight: how much VRAM+RAM the configured workers would donate,
and whether that covers an estimated model size.

I/O (ssh probes to each worker's node) lives here, not in core/vram.py, which
stays pure. `gather_donations`'s `gpus`/`ram` readers are dependency-injected
so the gather itself is unit-testable without any subprocess/ssh call.
"""
from pathlib import Path

from llama_launcher.core.nodes import LOCAL_NODE, connection_for
from llama_launcher.core.vram import pooled_fit
from llama_launcher.services import gpu as gpu_svc
from llama_launcher.services import sysstat
from llama_launcher.store.nodes import get_node

_MIB = 1024 * 1024
_GIB = 1024 ** 3


def _node_ssh_target(base_dir: Path, node_name: str) -> str:
    """'' for the local node (or an unknown one -- treated as local), else the
    saved ssh_target for a registered remote node."""
    node = get_node(base_dir, node_name) or LOCAL_NODE
    return node.ssh_target if node.kind == "remote" else ""


def free_vram_bytes(ssh_target: str = "") -> int:
    """Combined free VRAM across every GPU visible on that host (local when
    ssh_target is empty), via `nvidia-smi` (local or over ssh)."""
    return sum(g.mem_free_mib for g in gpu_svc.query_gpus(ssh_target)) * _MIB


def free_ram_bytes(ssh_target: str = "") -> int:
    """Free system RAM (MemAvailable, or MemTotal - used) on that host."""
    if ssh_target:
        mem = sysstat.read_remote_meminfo(ssh_target)
    else:
        _stat, mem_text, _load = sysstat.read_system()
        mem = sysstat.parse_meminfo(mem_text) if mem_text is not None else None
    if mem is None:
        return 0
    return max(0, mem.total_bytes - mem.used_bytes)


def default_gpus_reader(base_dir: Path):
    """A `gpus(node_name) -> free_bytes` reader bound to `base_dir`'s node
    registry: resolves the worker's node to its ssh_target (or local) and
    probes free VRAM there."""
    def _reader(node_name: str) -> int:
        return free_vram_bytes(_node_ssh_target(base_dir, node_name))
    return _reader


def default_ram_reader(base_dir: Path):
    """Same as `default_gpus_reader`, but probes free system RAM."""
    def _reader(node_name: str) -> int:
        return free_ram_bytes(_node_ssh_target(base_dir, node_name))
    return _reader


def gather_donations(profile, base_dir, *, gpus=None, ram=None) -> list:
    """One `(kind, bytes)` donation per configured RPC worker.

    A worker whose `device` starts with "CUDA" donates VRAM; any other device
    (e.g. "CPU") donates RAM. A pledged `--mem` budget (`worker.mem_mb > 0`)
    wins over probing; otherwise the injected `gpus`/`ram` reader supplies the
    live free-memory figure for that worker's node. `gpus`/`ram` default to
    real ssh/nvidia-smi/`/proc/meminfo` probes keyed off `base_dir`'s node
    registry (see `default_gpus_reader`/`default_ram_reader`).
    """
    gpus = gpus or default_gpus_reader(base_dir)
    ram = ram or default_ram_reader(base_dir)
    donations = []
    # A node's free VRAM (or free RAM) is one shared pool: two no-pledge workers
    # on the SAME node draw from it, so probing it once per worker double-counts
    # and inflates the fit. Read each node's free memory once per kind; an
    # explicit `--mem` pledge is a per-worker allocation the user chose, so
    # pledges stay additive.
    counted_free: set = set()
    for w in profile.runtime.rpc_workers:
        is_vram = (w.device or "").upper().startswith("CUDA")
        kind = "vram" if is_vram else "ram"
        if w.mem_mb:
            amount = int(w.mem_mb) * _MIB
        elif (w.node, kind) in counted_free:
            amount = 0
        else:
            counted_free.add((w.node, kind))
            amount = int(gpus(w.node)) if is_vram else int(ram(w.node))
        donations.append((kind, amount))
    return donations


def _gib(n: int) -> float:
    return n / _GIB


def headline(estimate_bytes: int, donations: list) -> str:
    """One human-readable line summarizing whether the pool fits the model,
    e.g. "120 GB model -> fits: 48 GB VRAM + 96 GB RAM = 144 GB (margin 24 GB)".
    """
    fit = pooled_fit(estimate_bytes, donations)
    verb = "fits" if fit.fits else "does not fit"
    tail = f"margin {_gib(fit.margin):.0f} GB" if fit.fits \
        else f"short by {_gib(-fit.margin):.0f} GB"
    return (f"{_gib(estimate_bytes):.0f} GB model -> {verb}: "
            f"{_gib(fit.vram_bytes):.0f} GB VRAM + {_gib(fit.ram_bytes):.0f} GB RAM "
            f"= {_gib(fit.total_bytes):.0f} GB ({tail})")
