import dataclasses
import re
from pathlib import Path

from llama_launcher.core.capabilities import derive_caps
from llama_launcher.core.gguf import GgufMeta, parse_gguf_header
from llama_launcher.core.pathmap import container_to_host

_SPLIT_RE = re.compile(r"^(?P<stem>.*)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$")


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


def split_parts(host_path, count=None) -> list[str]:
    """Every part name implied by a split model's part-of-total filename
    suffix, in part order, derived from the name alone with no filesystem
    access. count overrides the total taken from the name when given. A
    file whose part number exceeds the effective total, or that carries no
    such suffix, is its own only part."""
    p = Path(host_path)
    m = _SPLIT_RE.match(p.name)
    if m is None:
        return [str(p)]
    no = int(m.group("no"))
    total = count if count is not None else int(m.group("count"))
    if no > total:
        return [str(p)]
    stem = m.group("stem")
    return [
        str(p.with_name(f"{stem}-{i:05d}-of-{total:05d}.gguf"))
        for i in range(1, total + 1)
    ]


def read_model(host_path) -> tuple[GgufMeta | None, int | None]:
    """Metadata and size of a model file, with the tensor tables and sizes of
    every part merged when the file is one part of a split model. Parts are
    named from the file's own part-of-total suffix; when the meta's
    split.count key names a different total, the two namings are merged so
    a correctly named sibling is never dropped. A part that cannot be read
    contributes nothing."""
    meta = read_gguf_meta(host_path)
    size = file_size(host_path)
    if meta is None:
        return meta, size
    parts = split_parts(host_path)
    if meta.split_count > 1:
        keyed = split_parts(host_path, count=meta.split_count)
        seen: set[Path] = set()
        merged = []
        for part in (*parts, *keyed):
            resolved = Path(part)
            if resolved not in seen:
                seen.add(resolved)
                merged.append(part)
        parts = merged
    if len(parts) <= 1:
        return meta, size
    tensors = list(meta.tensors)
    total = size or 0
    for part in parts:
        if Path(part) == Path(host_path):
            continue
        # A part after the first carries only a small key-value block and
        # its tensor table; the small cap can cut that table short, so
        # fall back to the default-sized read whenever the small read
        # comes back empty or unreadable.
        part_meta = read_gguf_meta(part, max_bytes=8 * 1024 * 1024)
        if part_meta is None or not part_meta.tensors:
            part_meta = read_gguf_meta(part)
        if part_meta is None:
            continue
        tensors.extend(part_meta.tensors)
        total += file_size(part) or 0
    return dataclasses.replace(meta, tensors=tuple(tensors)), total


def inspect_file(container_path, mounts) -> tuple[GgufMeta | None, int | None]:
    """Metadata and size of any GGUF under a mount, for the draft model and
    the projector; (None, None) when the path is under no mount."""
    host = container_to_host(container_path, mounts)
    if host is None:
        return None, None
    return read_model(host)


def inspect_model(container_path, mounts):
    """Resolve container_path to the host, read meta and size (merging every
    part of a split model), derive caps. Returns (GgufMeta|None, int|None,
    ModelCaps|None); all-None if not under a mount."""
    host = container_to_host(container_path, mounts)
    if host is None:
        return None, None, None
    meta, size = read_model(host)
    return meta, size, derive_caps(meta, sibling_ggufs(host))
