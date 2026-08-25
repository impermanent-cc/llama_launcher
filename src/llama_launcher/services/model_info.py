from pathlib import Path

from llama_launcher.core.gguf import parse_gguf_header, GgufMeta
from llama_launcher.core.pathmap import container_to_host
from llama_launcher.core.capabilities import derive_caps, ModelCaps


def read_gguf_meta(path, max_bytes: int = 64 * 1024 * 1024) -> GgufMeta | None:
    # GGUF metadata (including the tokenizer vocab) lives at the file start,
    # before the tensor data. A large-vocab model can push the architecture keys
    # past a tight cap -> parse_gguf_header raises -> None -> vram falls back to a
    # weights-only (no KV) estimate silently. 64MB comfortably covers even
    # 256k-token vocabs while still reading only the header, not the weights.
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
        return parse_gguf_header(data)
    except (OSError, ValueError):
        return None


def file_size(path) -> int | None:
    try:
        return Path(path).stat().st_size
    except OSError:
        return None


def sibling_ggufs(host_model_path) -> list[str]:
    """GGUF filenames beside the model (in its directory and its parent),
    excluding the model file itself. Empty on any error."""
    p = Path(host_model_path)
    names: set[str] = set()
    for d in (p.parent, p.parent.parent):
        try:
            for f in d.iterdir():
                if f.suffix == ".gguf" and f.name != p.name:
                    names.add(f.name)
        except OSError:
            continue
    return sorted(names)


def inspect_model(container_path, mounts):
    """Resolve container_path to the host, read meta + size once, derive caps.
    Returns (GgufMeta|None, int|None, ModelCaps|None); all-None if not under a mount."""
    host = container_to_host(container_path, mounts)
    if host is None:
        return None, None, None
    meta = read_gguf_meta(host)
    return meta, file_size(host), derive_caps(meta, sibling_ggufs(host))
