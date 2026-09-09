from llama_launcher.core.sweep import MeasuredMemory, Sweep, SweepPoint
from llama_launcher.services import sweep_store


def _sweep(knob="n-cpu-ffn"):
    m = MeasuredMemory(
        ({"model": 1, "kv": 2, "compute": 3},),
        {"model": 0, "kv": 0, "compute": 0, "output": 4},
    )
    p = SweepPoint(
        2,
        "ok",
        10.5,
        (
            {
                "target_size": 128,
                "prompt_n": 128,
                "pp_tok_s": 1.0,
                "gen_tok_s": 2.0,
                "total_s": 3.0,
            },
        ),
        m,
        (100,),
        50,
        "",
    )
    return Sweep(
        "My Prof",
        knob,
        "2026-09-06T10:00:00",
        (p,),
        {"sizes": [128], "n_predict": 8, "warmup": 0, "repeats": 1},
    )


def test_save_and_load_round_trip(tmp_path):
    sweep_store.save(tmp_path, "My Prof", _sweep())
    d = sweep_store.load(tmp_path, "My Prof")
    assert d["knob"] == "n-cpu-ffn" and d["points"][0]["count"] == 2
    assert d["points"][0]["measured"]["cards"][0]["compute"] == 3
    assert sweep_store.sweep_path(tmp_path, "My Prof").name == "my-prof.json"


def test_load_missing_or_corrupt_is_none(tmp_path):
    assert sweep_store.load(tmp_path, "nope") is None
    p = sweep_store.sweep_path(tmp_path, "bad")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json")
    assert sweep_store.load(tmp_path, "bad") is None


def test_save_creates_the_sweeps_dir_with_a_private_file(tmp_path):
    sweep_store.save(tmp_path, "P", _sweep())
    p = sweep_store.sweep_path(tmp_path, "P")
    assert p.parent.name == "sweeps" and p.parent.is_dir()
    assert oct(p.stat().st_mode & 0o777) == "0o600"


def test_a_second_save_replaces_the_first(tmp_path):
    sweep_store.save(tmp_path, "P", _sweep(knob="n-cpu-ffn"))
    sweep_store.save(tmp_path, "P", _sweep(knob="n-cpu-moe"))
    assert sweep_store.load(tmp_path, "P")["knob"] == "n-cpu-moe"
