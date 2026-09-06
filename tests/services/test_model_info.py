import os
import struct
from pathlib import Path

import pytest

from llama_launcher.core.gguf import GgufMeta, TensorInfo
from llama_launcher.core.spec import Mount
from llama_launcher.services import model_info
from llama_launcher.services.model_info import (
    file_size,
    inspect_model,
    read_gguf_meta,
    sibling_ggufs,
)


def _fake_meta(names, split_count=1):
    return GgufMeta(
        arch="llama",
        n_layers=2,
        split_count=split_count,
        tensors=tuple(TensorInfo(n, 1, 0, 4) for n in names),
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


def test_inspect_model_sums_every_split_part(tmp_path, monkeypatch):
    p1 = tmp_path / "m-00001-of-00002.gguf"
    p2 = tmp_path / "m-00002-of-00002.gguf"
    p1.write_bytes(b"a" * 10)
    p2.write_bytes(b"b" * 30)
    metas = {
        str(p1): _fake_meta(["blk.0.attn_q.weight"], split_count=2),
        str(p2): _fake_meta(["blk.1.attn_q.weight"], split_count=2),
    }
    monkeypatch.setattr(
        model_info, "read_gguf_meta", lambda path, **kw: metas.get(str(path))
    )
    mounts = [Mount(host=str(tmp_path), container="/models", role="model")]
    meta, size, caps = model_info.inspect_model("/models/m-00001-of-00002.gguf", mounts)
    assert size == 40
    assert [t.name for t in meta.tensors] == [
        "blk.0.attn_q.weight",
        "blk.1.attn_q.weight",
    ]
    assert caps is not None


def test_inspect_model_not_under_mount(tmp_path):
    meta, size, caps = inspect_model(
        "/elsewhere/m.gguf", [Mount(host=str(tmp_path), container="/models")]
    )
    assert (meta, size, caps) == (None, None, None)


def test_split_parts_lists_every_part_in_order(tmp_path):
    parts = model_info.split_parts(str(tmp_path / "m-00002-of-00003.gguf"))
    assert [os.path.basename(p) for p in parts] == [
        "m-00001-of-00003.gguf",
        "m-00002-of-00003.gguf",
        "m-00003-of-00003.gguf",
    ]


def test_split_parts_single_file_is_itself(tmp_path):
    f = tmp_path / "m.gguf"
    assert model_info.split_parts(str(f)) == [str(f)]


def test_split_parts_number_exceeding_count_is_its_own_part(tmp_path):
    f = tmp_path / "m-00004-of-00002.gguf"
    assert model_info.split_parts(str(f)) == [str(f)]


def test_read_model_merges_split_tensor_tables(tmp_path, monkeypatch):
    p1 = tmp_path / "m-00001-of-00002.gguf"
    p2 = tmp_path / "m-00002-of-00002.gguf"
    p1.write_bytes(b"a" * 10)
    p2.write_bytes(b"b" * 30)
    metas = {
        str(p1): _fake_meta(["blk.0.attn_q.weight"], split_count=2),
        str(p2): _fake_meta(["blk.1.attn_q.weight"], split_count=2),
    }
    monkeypatch.setattr(
        model_info, "read_gguf_meta", lambda path, **kw: metas[str(path)]
    )
    meta, size = model_info.read_model(str(p1))
    assert [t.name for t in meta.tensors] == [
        "blk.0.attn_q.weight",
        "blk.1.attn_q.weight",
    ]
    assert size == 40


def test_read_model_merges_when_split_count_key_is_absent(tmp_path, monkeypatch):
    p1 = tmp_path / "m-00001-of-00002.gguf"
    p2 = tmp_path / "m-00002-of-00002.gguf"
    p1.write_bytes(b"a" * 10)
    p2.write_bytes(b"b" * 20)
    metas = {
        str(p1): _fake_meta(["blk.0.attn_q.weight"]),
        str(p2): _fake_meta(["blk.1.attn_q.weight"]),
    }
    monkeypatch.setattr(
        model_info, "read_gguf_meta", lambda path, **kw: metas.get(str(path))
    )
    meta, size = model_info.read_model(str(p1))
    assert [t.name for t in meta.tensors] == [
        "blk.0.attn_q.weight",
        "blk.1.attn_q.weight",
    ]
    assert size == 30


def test_read_model_key_larger_than_name_includes_extra_part(tmp_path, monkeypatch):
    p1 = tmp_path / "m-00001-of-00002.gguf"
    p1.write_bytes(b"a" * 10)
    extra = tmp_path / "m-00003-of-00003.gguf"
    metas = {
        str(p1): _fake_meta(["blk.0.attn_q.weight"], split_count=3),
        str(extra): _fake_meta(["blk.2.attn_q.weight"], split_count=3),
    }
    monkeypatch.setattr(
        model_info, "read_gguf_meta", lambda path, **kw: metas.get(str(path))
    )
    meta, _size = model_info.read_model(str(p1))
    assert [t.name for t in meta.tensors] == [
        "blk.0.attn_q.weight",
        "blk.2.attn_q.weight",
    ]


def test_read_model_key_disagreement_unions_both_namings(tmp_path, monkeypatch):
    p1 = tmp_path / "m-00001-of-00002.gguf"
    p2 = tmp_path / "m-00002-of-00002.gguf"
    p3 = tmp_path / "m-00003-of-00003.gguf"
    p1.write_bytes(b"a" * 10)
    p2.write_bytes(b"b" * 20)
    p3.write_bytes(b"c" * 30)
    metas = {
        str(p1): _fake_meta(["blk.0.attn_q.weight"], split_count=3),
        str(p2): _fake_meta(["blk.1.attn_q.weight"], split_count=3),
        str(p3): _fake_meta(["blk.2.attn_q.weight"], split_count=3),
    }
    monkeypatch.setattr(
        model_info, "read_gguf_meta", lambda path, **kw: metas.get(str(path))
    )
    meta, size = model_info.read_model(str(p1))
    assert [t.name for t in meta.tensors] == [
        "blk.0.attn_q.weight",
        "blk.1.attn_q.weight",
        "blk.2.attn_q.weight",
    ]
    assert size == 60


def test_split_parts_clamps_when_count_is_overridden(tmp_path):
    f = tmp_path / "m-00004-of-00002.gguf"
    assert model_info.split_parts(str(f), count=2) == [str(f)]


def test_read_model_falls_back_when_small_read_has_no_tensors(tmp_path, monkeypatch):
    p1 = tmp_path / "m-00001-of-00002.gguf"
    p2 = tmp_path / "m-00002-of-00002.gguf"
    p1.write_bytes(b"a" * 10)
    p2.write_bytes(b"b" * 20)
    host_meta = _fake_meta(["blk.0.attn_q.weight"], split_count=2)
    truncated = _fake_meta([], split_count=2)
    full = _fake_meta(["blk.1.attn_q.weight"], split_count=2)

    def fake_read(path, max_bytes=64 * 1024 * 1024):
        if str(path) == str(p1):
            return host_meta
        return truncated if max_bytes == 8 * 1024 * 1024 else full

    monkeypatch.setattr(model_info, "read_gguf_meta", fake_read)
    meta, size = model_info.read_model(str(p1))
    assert [t.name for t in meta.tensors] == [
        "blk.0.attn_q.weight",
        "blk.1.attn_q.weight",
    ]
    assert size == 30


def test_read_model_unreadable_part_keeps_first_part(tmp_path, monkeypatch):
    p1 = tmp_path / "m-00001-of-00002.gguf"
    p1.write_bytes(b"a" * 10)
    metas = {str(p1): _fake_meta(["blk.0.attn_q.weight"], split_count=2)}
    monkeypatch.setattr(
        model_info, "read_gguf_meta", lambda path, **kw: metas.get(str(path))
    )
    meta, size = model_info.read_model(str(p1))
    assert [t.name for t in meta.tensors] == ["blk.0.attn_q.weight"]
    assert size == 10


def test_inspect_file_resolves_under_mount(tmp_path, monkeypatch):
    f = tmp_path / "d.gguf"
    f.write_bytes(b"x" * 7)
    monkeypatch.setattr(model_info, "read_gguf_meta", lambda path, **kw: _fake_meta([]))
    mounts = [Mount(host=str(tmp_path), container="/models", role="model")]
    meta, size = model_info.inspect_file("/models/d.gguf", mounts)
    assert meta is not None and size == 7


def test_inspect_file_outside_every_mount(tmp_path):
    mounts = [Mount(host=str(tmp_path), container="/models", role="model")]
    assert model_info.inspect_file("/elsewhere/d.gguf", mounts) == (None, None)


@pytest.mark.skipif(
    not os.environ.get("LLAMA_LAUNCHER_TEST_GGUF"),
    reason="LLAMA_LAUNCHER_TEST_GGUF unset",
)
def test_real_gguf_has_a_tensor_table():
    meta, size = model_info.read_model(os.environ["LLAMA_LAUNCHER_TEST_GGUF"])
    assert meta is not None and size
    assert meta.tensors
    assert any(t.name.startswith("blk.0.") for t in meta.tensors)
    assert sum(t.nbytes for t in meta.tensors) <= size
