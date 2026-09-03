from dataclasses import asdict

from llama_launcher.services import benchmark_store as store
from llama_launcher.services.benchmark import BenchmarkRow, BenchmarkRun


def _run(ts, pp, gen, size=512):
    return BenchmarkRun(
        ts, [size], 128, 1, 3, [BenchmarkRow(size, size, pp, gen, 1.0)], {"model": "m"}
    )


def test_append_caps_at_five(tmp_path):
    for i in range(7):
        got = store.append(tmp_path, "prof", _run(f"t{i}", 100 + i, 50))
    assert len(got) == 5
    assert [r["timestamp"] for r in got] == [
        "t2",
        "t3",
        "t4",
        "t5",
        "t6",
    ]  # oldest dropped


def test_load_roundtrip_and_missing(tmp_path):
    assert store.load(tmp_path, "nope") == []
    store.append(tmp_path, "prof", _run("t0", 100, 50))
    assert store.load(tmp_path, "prof")[0]["rows"][0]["pp_tok_s"] == 100


def test_clear_removes_history(tmp_path):
    store.append(tmp_path, "prof", _run("t0", 100, 50))
    assert store.load(tmp_path, "prof")  # non-empty
    store.clear(tmp_path, "prof")
    assert store.load(tmp_path, "prof") == []


def test_clear_is_safe_when_no_history(tmp_path):
    store.clear(tmp_path, "never-benched")  # must not raise
    assert store.load(tmp_path, "never-benched") == []


def test_delta_percent_and_size_mismatch():
    new = asdict(_run("n", 200, 60, size=512))
    old = asdict(_run("o", 100, 50, size=512))
    d = store.delta(new, old)
    assert d["sizes_differ"] is False
    assert abs(d["shared"][0]["pp_pct"] - 100.0) < 1e-6  # 100 to 200 = +100%
    assert abs(d["shared"][0]["gen_pct"] - 20.0) < 1e-6  # 50 to 60   = +20%
    d2 = store.delta(asdict(_run("n", 200, 60, size=256)), old)
    assert d2["sizes_differ"] is True
