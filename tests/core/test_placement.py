import pathlib

import pytest

from llama_launcher.core import placement as pl
from llama_launcher.core.gguf import TensorInfo

FIXTURE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "fixtures"
    / "upstream_tensor_patterns.txt"
)


def _tensors(n_layers=4, moe=False):
    out = [
        TensorInfo("token_embd.weight", 1, 0, 100),
        TensorInfo("rope_freqs.weight", 1, 0, 1),
    ]
    for i in range(n_layers):
        out.append(TensorInfo(f"blk.{i}.attn_q.weight", 1, 0, 10))
        out.append(TensorInfo(f"blk.{i}.attn_norm.weight", 1, 0, 1))
        if moe:
            out.append(TensorInfo(f"blk.{i}.ffn_up_exps.weight", 1, 0, 40))
            out.append(TensorInfo(f"blk.{i}.ffn_gate_inp.weight", 1, 0, 2))
        else:
            out.append(TensorInfo(f"blk.{i}.ffn_up.weight", 1, 0, 20))
            out.append(TensorInfo(f"blk.{i}.ffn_down.weight", 1, 0, 20))
    out.append(TensorInfo("output_norm.weight", 1, 0, 1))
    out.append(TensorInfo("output.weight", 1, 0, 50))
    return out


def test_patterns_match_upstream_fixture():
    lines = [
        ln for ln in FIXTURE.read_text().splitlines() if ln and not ln.startswith("#")
    ]
    assert lines == [pl.EXPS_REGEX, pl.DENSE_FFN_REGEX, pl.ALL_EXPS_PATTERN]


@pytest.mark.parametrize(
    "value,engine,expected",
    [
        ("auto", "llama.cpp", 5),
        ("all", "llama.cpp", 5),
        ("auto", "ik_llama.cpp", 0),
        ("all", "ik_llama.cpp", 5),
        (2, "llama.cpp", 2),
        ("2", "llama.cpp", 2),
        (99, "llama.cpp", 5),
        (-1, "llama.cpp", 5),
        (-1, "ik_llama.cpp", 0),
        (None, "llama.cpp", 5),
        ("junk", "llama.cpp", 5),
    ],
)
def test_gpu_layer_count(value, engine, expected):
    assert pl.gpu_layer_count(value, 4, engine) == expected


def test_layer_devices_mainline_offloads_output_first():
    # mainline: ngl 1 puts the output layer on the GPU and every block on CPU
    assert pl.layer_devices(4, 1, "llama.cpp") == (False, False, False, False, True)
    assert pl.layer_devices(4, 3, "llama.cpp") == (False, False, True, True, True)
    assert pl.layer_devices(4, 5, "llama.cpp") == (True,) * 5
    assert pl.layer_devices(4, 0, "llama.cpp") == (False,) * 5


def test_layer_devices_ik_counts_blocks_then_output():
    assert pl.layer_devices(4, 1, "ik_llama.cpp") == (False, False, False, True, False)
    assert pl.layer_devices(4, 4, "ik_llama.cpp") == (True, True, True, True, False)
    assert pl.layer_devices(4, 5, "ik_llama.cpp") == (True,) * 5


def test_cpu_rules_from_flags_respect_engine_gate():
    rules = pl.cpu_rules({"n-cpu-moe": 2, "n-cpu-ffn": 3, "cpu-moe": True}, "llama.cpp")
    assert (None, pl.EXPS_REGEX) in [(r.max_layer, r.regex) for r in rules]
    assert (2, pl.EXPS_REGEX) in [(r.max_layer, r.regex) for r in rules]
    assert (3, pl.DENSE_FFN_REGEX) in [(r.max_layer, r.regex) for r in rules]
    ik = pl.cpu_rules({"n-cpu-moe": 2, "n-cpu-ffn": 3}, "ik_llama.cpp")
    assert [(r.max_layer, r.regex) for r in ik] == [(2, pl.EXPS_REGEX)]


def test_cpu_rules_draft_reads_draft_twins_only():
    s = {
        "n-cpu-moe": 2,
        "spec-draft-n-cpu-moe": 1,
        "spec-draft-cpu-moe": True,
        "n-cpu-ffn": 3,
    }
    draft = pl.cpu_rules(s, "llama.cpp", draft=True)
    assert [(r.max_layer, r.regex) for r in draft] == [
        (None, pl.EXPS_REGEX),
        (1, pl.EXPS_REGEX),
    ]
    assert pl.cpu_rules(s, "ik_llama.cpp", draft=True) == []


def test_parse_overrides_first_match_wins_and_bad_regex_is_dropped():
    rules = pl.parse_overrides(r"exps=CPU, blk\.3\.=CUDA1,( =CPU,novalue")
    assert rules == [("exps", True), (r"blk\.3\.", False)]
    assert pl.parse_overrides("") == []
    assert pl.parse_overrides(None) == []


def test_place_dense_all_gpu():
    t = _tensors()
    p = pl.place(
        t, 4, devices=pl.layer_devices(4, 5, "llama.cpp"), rules=[], overrides=[]
    )
    assert p.input_cpu == 101
    assert p.output_gpu == 51 and p.output_cpu == 0
    assert p.layer_gpu == (51, 51, 51, 51) and p.layer_cpu == (0, 0, 0, 0)
    assert p.gpu_bytes == 4 * 51 + 51
    assert p.cpu_bytes == 101


def test_place_n_cpu_ffn_moves_dense_ffn_of_first_layers():
    t = _tensors()
    p = pl.place(
        t,
        4,
        devices=pl.layer_devices(4, 5, "llama.cpp"),
        rules=pl.cpu_rules({"n-cpu-ffn": 2}, "llama.cpp"),
        overrides=[],
    )
    assert p.layer_gpu == (11, 11, 51, 51)
    assert p.layer_cpu == (40, 40, 0, 0)


def test_place_moe_cpu_moe_keeps_router_and_attention_on_gpu():
    t = _tensors(moe=True)
    p = pl.place(
        t,
        4,
        devices=pl.layer_devices(4, 5, "llama.cpp"),
        rules=pl.cpu_rules({"cpu-moe": True}, "llama.cpp"),
        overrides=[],
    )
    assert p.layer_gpu == (13, 13, 13, 13)
    assert p.layer_cpu == (40, 40, 40, 40)


def test_place_partial_ngl_puts_low_layers_and_their_ffn_on_cpu():
    t = _tensors()
    p = pl.place(
        t, 4, devices=pl.layer_devices(4, 3, "llama.cpp"), rules=[], overrides=[]
    )
    assert p.layer_gpu == (0, 0, 51, 51)
    assert p.layer_cpu == (51, 51, 0, 0)
    assert p.output_gpu == 51


def test_place_override_to_cpu_and_back_to_gpu_first_match_wins():
    t = _tensors()
    p = pl.place(
        t,
        4,
        devices=pl.layer_devices(4, 5, "llama.cpp"),
        rules=[],
        overrides=pl.parse_overrides(r"blk\.1\.ffn_up=CUDA0,ffn_up=CPU"),
    )
    assert p.layer_cpu == (20, 0, 20, 20)


def test_place_override_moves_a_ram_layer_onto_a_card():
    t = _tensors()
    p = pl.place(
        t,
        4,
        devices=pl.layer_devices(4, 3, "llama.cpp"),
        rules=[],
        overrides=pl.parse_overrides(r"blk\.1\.=CUDA0"),
    )
    assert p.layer_gpu == (0, 51, 51, 51)
    assert p.layer_cpu == (51, 0, 0, 0)


def test_place_override_does_not_reclaim_a_tensor_a_cpu_rule_took():
    t = _tensors()
    p = pl.place(
        t,
        4,
        devices=pl.layer_devices(4, 5, "llama.cpp"),
        rules=pl.cpu_rules({"n-cpu-ffn": 2}, "llama.cpp"),
        overrides=pl.parse_overrides(r"blk\.0\.ffn_up=CUDA0"),
    )
    assert p.layer_cpu == (40, 40, 0, 0)
    assert p.layer_gpu == (11, 11, 51, 51)


def test_place_fallback_without_tensor_table():
    p = pl.place(
        (),
        4,
        devices=pl.layer_devices(4, 5, "llama.cpp"),
        rules=[],
        overrides=[],
        weights_fallback=1000,
    )
    assert p.gpu_bytes == 1000 and p.cpu_bytes == 0
    p = pl.place(
        (),
        4,
        devices=pl.layer_devices(4, 0, "ik_llama.cpp"),
        rules=[],
        overrides=[],
        weights_fallback=1000,
    )
    assert p.gpu_bytes == 0 and p.cpu_bytes == 1000


def test_place_fallback_counts_any_gpu_layer_under_ik():
    """Under ik_llama.cpp, --n-gpu-layers equal to the layer count keeps
    the output layer in RAM but every block layer is on a card: without a
    tensor table the fallback still counts as GPU weights."""
    p = pl.place(
        (),
        4,
        devices=pl.layer_devices(4, 4, "ik_llama.cpp"),
        rules=[],
        overrides=[],
        weights_fallback=1000,
    )
    assert p.gpu_bytes == 1000 and p.cpu_bytes == 0


def test_place_tied_embeddings_duplicate_token_embd_at_output_layer():
    t = [
        TensorInfo("token_embd.weight", 1, 0, 100),
        TensorInfo("blk.0.attn_q.weight", 1, 0, 10),
    ]
    on_gpu = pl.place(t, 1, devices=(True, True), rules=[], overrides=[])
    assert on_gpu.output_gpu == 100 and on_gpu.output_cpu == 0
    on_cpu = pl.place(t, 1, devices=(True, False), rules=[], overrides=[])
    assert on_cpu.output_cpu == 100 and on_cpu.output_gpu == 0


def test_place_invalid_raw_override_pattern_is_skipped_not_raised():
    t = _tensors()
    p = pl.place(
        t,
        4,
        devices=pl.layer_devices(4, 5, "llama.cpp"),
        rules=[],
        overrides=[("(", True)],
    )
    assert p.gpu_bytes == 4 * 51 + 51


def test_split_fractions_explicit_then_free_then_equal():
    assert pl.split_fractions("60,40", [1, 1], 2) == [0.6, 1.0]
    assert pl.split_fractions("3/1", [1, 1], 2) == [0.75, 1.0]
    assert pl.split_fractions("", [16, 8], 2) == [16 / 24, 1.0]
    assert pl.split_fractions("", [0, 0], 2) == [0.5, 1.0]
    assert pl.split_fractions("1", [1, 1], 2) == [1.0, 1.0]
    assert pl.split_fractions("x,y", [1, 3], 2) == [0.25, 1.0]


def test_split_fractions_none_entry_counts_as_zero():
    assert pl.split_fractions("", [None, 8], 2) == [0.0, 1.0]


def test_distribute_layer_mode_follows_upstream_fractions():
    devices = pl.layer_devices(4, 5, "llama.cpp")
    p = pl.place(_tensors(), 4, devices=devices, rules=[], overrides=[])
    d = pl.distribute(
        p,
        free_per_card=[1, 1],
        tensor_split="50,50",
        split_mode="layer",
        main_gpu=0,
        engine="llama.cpp",
    )
    # fractions (il - start) / act: 0, .2, .4, .6, .8 ; upper_bound over [.5, 1.0]
    assert d.weight_share == (
        (1.0, 0.0),
        (1.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.0, 1.0),
    )
    assert d.kv_card == (0, 0, 0, 1)


def test_distribute_cpu_layers_have_no_card():
    devices = pl.layer_devices(4, 3, "llama.cpp")
    p = pl.place(_tensors(), 4, devices=devices, rules=[], overrides=[])
    d = pl.distribute(
        p,
        free_per_card=[1],
        tensor_split="",
        split_mode="layer",
        main_gpu=0,
        engine="llama.cpp",
    )
    assert d.weight_share[0] == (0.0,) and d.kv_card[0] is None
    assert d.weight_share[4] == (1.0,)


def test_distribute_none_and_row_modes():
    devices = pl.layer_devices(4, 5, "llama.cpp")
    p = pl.place(_tensors(), 4, devices=devices, rules=[], overrides=[])
    none = pl.distribute(
        p,
        free_per_card=[1, 1],
        tensor_split="",
        split_mode="none",
        main_gpu=1,
        engine="llama.cpp",
    )
    assert all(s == (0.0, 1.0) for s in none.weight_share)
    assert none.kv_card == (1, 1, 1, 1)
    row = pl.distribute(
        p,
        free_per_card=[1, 1],
        tensor_split="75,25",
        split_mode="row",
        main_gpu=1,
        engine="llama.cpp",
    )
    assert all(s == (0.75, 0.25) for s in row.weight_share)
    assert row.kv_card == (1, 1, 1, 1)


def test_distribute_none_mode_converts_string_main_gpu():
    devices = pl.layer_devices(4, 5, "llama.cpp")
    p = pl.place(_tensors(), 4, devices=devices, rules=[], overrides=[])
    d = pl.distribute(
        p,
        free_per_card=[1, 1],
        tensor_split="",
        split_mode="none",
        main_gpu="1",
        engine="llama.cpp",
    )
    assert all(s == (0.0, 1.0) for s in d.weight_share)
    assert d.kv_card == (1, 1, 1, 1)


def test_distribute_row_mode_converts_string_main_gpu():
    devices = pl.layer_devices(4, 5, "llama.cpp")
    p = pl.place(_tensors(), 4, devices=devices, rules=[], overrides=[])
    d = pl.distribute(
        p,
        free_per_card=[1, 1],
        tensor_split="75,25",
        split_mode="row",
        main_gpu="1",
        engine="llama.cpp",
    )
    assert all(s == (0.75, 0.25) for s in d.weight_share)
    assert d.kv_card == (1, 1, 1, 1)


def test_distribute_promotes_an_override_layer_onto_a_card():
    """A layer --n-gpu-layers keeps in RAM but an override moves onto a
    card gets a card share (a unit vector), even though it falls outside
    the contiguous run --n-gpu-layers put on the cards; its KV cache stays
    in RAM since the override names weights, not the KV cache."""
    devices = pl.layer_devices(4, 3, "llama.cpp")
    p = pl.place(
        _tensors(),
        4,
        devices=devices,
        rules=[],
        overrides=pl.parse_overrides(r"blk\.1\.=CUDA0"),
    )
    d = pl.distribute(
        p,
        free_per_card=[1, 1],
        tensor_split="50,50",
        split_mode="layer",
        main_gpu=0,
        engine="llama.cpp",
    )
    assert d.weight_share[1] == (1.0, 0.0)
    assert d.kv_card[1] is None


def test_effective_settings_overlays_raw_args():
    eff = pl.effective_settings(
        {"ctx-size": 4096, "n-cpu-ffn": 2}, "--n-cpu-ffn 8 -ts 60,40"
    )
    assert (
        eff["n-cpu-ffn"] == "8"
        and eff["tensor-split"] == "60,40"
        and eff["ctx-size"] == 4096
    )


def test_effective_settings_joins_override_tensor_with_the_profile_value_first():
    """--override-tensor accumulates in argv, so the effective estimate must
    see the profile's own value ahead of a raw -ot entry, matching argv
    order, rather than the raw value replacing it."""
    eff = pl.effective_settings({"override-tensor": "p=CPU"}, "-ot a=CPU")
    assert eff["override-tensor"] == "p=CPU,a=CPU"


def test_effective_settings_override_tensor_raw_alone_when_profile_blank():
    for profile_settings in ({}, {"override-tensor": ""}, {"override-tensor": "  "}):
        eff = pl.effective_settings(profile_settings, "-ot a=CPU")
        assert eff["override-tensor"] == "a=CPU"


def test_effective_settings_non_accumulating_key_still_replaces():
    eff = pl.effective_settings({"n-cpu-ffn": 2}, "--n-cpu-ffn 8")
    assert eff["n-cpu-ffn"] == "8"
