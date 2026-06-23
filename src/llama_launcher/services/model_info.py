from pathlib import Path

from llama_launcher.core.gguf import parse_gguf_header, GgufMeta


def read_gguf_meta(path, max_bytes: int = 16 * 1024 * 1024) -> GgufMeta | None:
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
