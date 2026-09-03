import struct
from pathlib import Path

from llama_launcher.core.spec import Mount
from llama_launcher.services.model_info import (
    file_size,
    inspect_model,
    read_gguf_meta,
    sibling_ggufs,
)


def _write_gguf(path):
    def kv_str(k, v):
        kb, vb = k.encode(), v.encode()
        return (
            struct.pack("<Q", len(kb))
            + kb
            + struct.pack("<I", 8)
            + struct.pack("<Q", len(vb))
            + vb
        )

    def kv_u32(k, v):
        kb = k.encode()
        return (
            struct.pack("<Q", len(kb))
            + kb
            + struct.pack("<I", 4)
            + struct.pack("<I", v)
        )

    kvs = [kv_str("general.architecture", "llama"), kv_u32("llama.block_count", 32)]
    blob = (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", len(kvs))
        + b"".join(kvs)
    )
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


def test_sibling_ggufs_dir_and_parent(tmp_path):
    sub = tmp_path / "mtp"
    sub.mkdir()
    model = sub / "model.gguf"
    model.write_bytes(b"x")
    (sub / "other.gguf").write_bytes(b"x")  # same dir
    (tmp_path / "mmproj-F16.gguf").write_bytes(b"x")  # parent dir
    names = sibling_ggufs(model)
    assert "other.gguf" in names
    assert "mmproj-F16.gguf" in names
    assert "model.gguf" not in names  # excludes the model itself


def test_inspect_model_resolves_host(tmp_path):
    _write_gguf(tmp_path / "m.gguf")  # arch=llama in helper
    mounts = [Mount(host=str(tmp_path), container="/models")]
    meta, size, caps = inspect_model("/models/m.gguf", mounts)
    assert meta is not None and meta.arch == "llama"
    assert size and size > 0
    assert caps is not None


def test_inspect_model_not_under_mount(tmp_path):
    meta, size, caps = inspect_model(
        "/elsewhere/m.gguf", [Mount(host=str(tmp_path), container="/models")]
    )
    assert (meta, size, caps) == (None, None, None)
