import struct
from pathlib import Path
from llama_launcher.services.model_info import read_gguf_meta, file_size


def _write_gguf(path):
    def kv_str(k, v):
        kb, vb = k.encode(), v.encode()
        return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb
    def kv_u32(k, v):
        kb = k.encode()
        return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 4) + struct.pack("<I", v)
    kvs = [kv_str("general.architecture", "llama"), kv_u32("llama.block_count", 32)]
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs)) + b"".join(kvs)
    Path(path).write_bytes(blob)


def test_read_gguf_meta(tmp_path):
    p = tmp_path / "m.gguf"
    _write_gguf(p)
    m = read_gguf_meta(p)
    assert m is not None and m.arch == "llama" and m.n_layers == 32


def test_read_missing_returns_none(tmp_path):
    assert read_gguf_meta(tmp_path / "nope.gguf") is None


def test_read_garbage_returns_none(tmp_path):
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"not a gguf file at all")
    assert read_gguf_meta(p) is None


def test_file_size(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"12345")
    assert file_size(p) == 5
    assert file_size(tmp_path / "missing") is None
