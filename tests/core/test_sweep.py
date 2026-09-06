from llama_launcher.core import sweep as sw

LOG = """\
load_tensors: offloading 35 repeating layers to GPU
load_tensors:        CUDA0 model buffer size = 17904.12 MiB
load_tensors:        CUDA1 model buffer size =  4021.50 MiB
load_tensors:   CPU_Mapped model buffer size =   512.00 MiB
llama_kv_cache:      CUDA0 KV buffer size =   384.00 MiB
llama_kv_cache:      CUDA1 KV buffer size =    96.00 MiB
llama_context:        CPU  output buffer size =     0.58 MiB
llama_context:      CUDA0 compute buffer size =   898.00 MiB
llama_context:      CUDA1 compute buffer size =   300.25 MiB
llama_context:  CUDA_Host compute buffer size =    24.01 MiB
main: server is listening on http://127.0.0.1:8080
"""
MIB = 1024 * 1024


def test_sweep_knob():
    assert sw.sweep_knob(True) == "n-cpu-moe"
    assert sw.sweep_knob(False) == "n-cpu-ffn"


def test_sweep_counts_inclusive_clamped_and_deduped():
    assert sw.sweep_counts(0, 8, 2, 40) == [0, 2, 4, 6, 8]
    assert sw.sweep_counts(36, 44, 2, 40) == [36, 38, 40]
    assert sw.sweep_counts(5, 5, 2, 40) == [5]
    assert sw.sweep_counts(4, 0, 2, 40) == [4]
    assert sw.sweep_counts(0, 8, 0, 40) == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert sw.sweep_counts(0, 8, 2, None) == [0, 2, 4, 6, 8]


def test_default_range():
    assert sw.default_range(None) == (0, 8, 2)
    assert sw.default_range(0) == (0, 8, 2)
    assert sw.default_range(12) == (12, 20, 2)


def test_parse_load_log_maps_devices():
    m = sw.parse_load_log(LOG)
    assert m.cards[0] == {
        "model": int(17904.12 * MIB),
        "kv": 384 * MIB,
        "compute": 898 * MIB,
    }
    assert m.cards[1] == {
        "model": int(4021.5 * MIB),
        "kv": 96 * MIB,
        "compute": int(300.25 * MIB),
    }
    assert m.ram["model"] == 512 * MIB
    assert m.ram["compute"] == int(24.01 * MIB)
    assert m.ram["output"] == int(0.58 * MIB)
    assert m.ram["kv"] == 0
    assert sw.measured_card_total(m, 0) == m.cards[0]["model"] + 384 * MIB + 898 * MIB
    assert sw.measured_card_total(m, 5) == 0
    assert sw.measured_ram_total(m) == 512 * MIB + int(24.01 * MIB) + int(0.58 * MIB)


def test_parse_load_log_empty_and_gapped_cards():
    m = sw.parse_load_log("")
    assert m.cards == () and m.ram == {"model": 0, "kv": 0, "compute": 0, "output": 0}
    m = sw.parse_load_log("llama_kv_cache:      CUDA2 KV buffer size =    10.00 MiB\n")
    assert len(m.cards) == 3 and m.cards[0] == {"model": 0, "kv": 0, "compute": 0}
    assert m.cards[2]["kv"] == 10 * MIB


TIMESTAMPED_LOG = """\
0.02.690.534 I load_tensors:   CPU_Mapped model buffer size =  2483.69 MiB
0.02.690.541 I load_tensors:   CPU_REPACK model buffer size =  1222.80 MiB
0.03.564.186 I llama_context:        CPU  output buffer size =     4.00 MiB
0.03.564.545 I llama_kv_cache:        CPU KV buffer size =    12.00 MiB
0.03.567.592 I llama_kv_cache:        CPU KV buffer size =    24.00 MiB
0.03.594.837 I sched_reserve:        CPU compute buffer size =   113.52 MiB
0.03.594.857 I sched_reserve: graph splits = 1
"""


def test_parse_load_log_reads_timestamp_and_level_prefixed_lines():
    """A server that prefixes every log line with a timestamp and a level
    letter still yields per-device buffer sizes."""
    m = sw.parse_load_log(TIMESTAMPED_LOG)
    assert m.cards == ()
    assert m.ram["model"] == int(2483.69 * MIB) + int(1222.80 * MIB)
    assert m.ram["kv"] == 36 * MIB
    assert m.ram["compute"] == int(113.52 * MIB)
    assert m.ram["output"] == 4 * MIB


def test_parse_load_log_reads_a_timestamp_prefixed_cuda_line():
    """A prefixed CUDA buffer line still maps to its card."""
    m = sw.parse_load_log(
        "0.03.5 I load_tensors:        CUDA0 model buffer size = 17904.12 MiB\n"
    )
    assert m.cards[0]["model"] == int(17904.12 * MIB)


def test_last_log_line_skips_blank_tail():
    assert sw.last_log_line("a\nb\n\n  \n") == "b"
    assert sw.last_log_line("") == ""


def _point(count, status, gen, size=2048):
    rows = (
        {
            "target_size": 128,
            "prompt_n": 128,
            "pp_tok_s": 900.0,
            "gen_tok_s": gen + 5,
            "total_s": 1.0,
        },
        {
            "target_size": size,
            "prompt_n": size,
            "pp_tok_s": 800.0,
            "gen_tok_s": gen,
            "total_s": 2.0,
        },
    )
    return sw.SweepPoint(count, status, 12.0, rows, None, (), 0, "")


def test_best_point_uses_gen_at_largest_size_among_ok():
    pts = [
        _point(0, "failed", 99.0),
        _point(2, "ok", 30.0),
        _point(4, "ok", 31.5),
        _point(6, "ok", 31.5),
    ]
    assert sw.best_point(pts).count == 4
    assert sw.best_point([_point(0, "failed", 1.0)]) is None
    assert sw.best_point([]) is None


def test_best_point_ranks_a_row_without_a_gen_rate_as_zero():
    """A persisted row missing gen_tok_s ranks as 0.0 instead of raising."""
    partial = sw.SweepPoint(
        0, "ok", 12.0, ({"target_size": 2048, "pp_tok_s": 800.0},), None, (), 0, ""
    )
    assert sw._gen_at_largest(partial) == 0.0
    assert sw.best_point([partial]).count == 0
    assert sw.best_point([partial, _point(2, "ok", 10.0)]).count == 2


def test_largest_row_takes_the_last_row_of_equal_sizes():
    """One tie rule for every reader of a point's rows: the largest prompt
    size, last row wins."""
    rows = (
        {"target_size": 128, "gen_tok_s": 1.0},
        {"target_size": 2048, "gen_tok_s": 2.0},
        {"target_size": 2048, "gen_tok_s": 3.0},
    )
    assert sw.largest_row(rows)["gen_tok_s"] == 3.0
    assert sw.largest_row(({"pp_tok_s": 5.0},))["pp_tok_s"] == 5.0
    assert sw.largest_row(()) is None
    assert sw.largest_row(None) is None
