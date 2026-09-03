from llama_launcher.core.mtp_stats import (
    SpecCounters,
    parse_draft_stats,
    sparkline,
    spec_counters,
    spec_delta,
)

REAL = (
    "draft acceptance = 0.62008 ( 1797 accepted /  2898 generated), "
    "mean acceptance length =  2.24, acceptance rate per position = (0.727, 0.513)"
)


def test_parses_real_line():
    d = parse_draft_stats(REAL)
    assert d is not None
    assert abs(d.acceptance - 0.62008) < 1e-6
    assert d.accepted == 1797 and d.generated == 2898
    assert abs(d.mean_len - 2.24) < 1e-6
    assert d.per_position == (0.727, 0.513)


def test_three_positions():
    line = (
        "draft acceptance = 0.5 ( 3 accepted / 6 generated), "
        "mean acceptance length = 1.5, acceptance rate per position = (0.8, 0.5, 0.2)"
    )
    assert parse_draft_stats(line).per_position == (0.8, 0.5, 0.2)


def test_returns_last_of_many():
    blob = "noise\n" + REAL + "\nmid\n" + REAL.replace("0.62008", "0.71000") + "\ntail"
    assert abs(parse_draft_stats(blob).acceptance - 0.71) < 1e-6


def test_no_match_returns_none():
    assert parse_draft_stats("just some unrelated log line\n") is None


def test_partial_line_returns_none():
    # cut off before the per-position group
    partial = "draft acceptance = 0.6 ( 1 accepted / 2 generated), mean acceptance length = 1.2"
    assert parse_draft_stats(partial) is None


def test_sparkline_empty():
    assert sparkline([]) == ""


def test_sparkline_single_value():
    assert sparkline([5.0]) == "\u2581"


def test_sparkline_flat_series_all_low():
    assert sparkline([5, 5, 5]) == "\u2581\u2581\u2581"


def test_sparkline_ascending_ramp():
    s = sparkline([1, 2, 3, 4, 5, 6, 7, 8])
    assert len(s) == 8
    assert s[0] == "\u2581" and s[-1] == "\u2588"
    # monotonically non-decreasing block heights
    assert list(s) == sorted(
        s, key="\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588".index
    )


def test_sparkline_width_keeps_last_n():
    assert sparkline([1, 2, 3, 4, 5], width=2) == "\u2581\u2588"


METRICS = """
llamacpp:spec_decode_num_draft_tokens_total 1000
llamacpp:spec_decode_num_accepted_tokens_total 600
llamacpp:spec_decode_num_drafts_total 300
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 250
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 150
"""


def test_spec_counters_absent_when_spec_decode_off():
    assert spec_counters("llamacpp:n_decode_total 5\n") is None


def test_spec_counters_reads_all_three_totals():
    c = spec_counters(METRICS)
    assert c.draft_tokens == 1000
    assert c.accepted == 600
    assert c.drafts == 300
    assert c.per_position == (250.0, 150.0)


def test_spec_delta_derives_acceptance_and_mean_length():
    prev = SpecCounters(
        draft_tokens=1000, accepted=600, drafts=300, per_position=(250.0,)
    )
    cur = SpecCounters(
        draft_tokens=2000, accepted=1400, drafts=700, per_position=(600.0,)
    )
    stats = spec_delta(prev, cur)
    # 800 accepted of 1000 drafted since the last poll
    assert stats.acceptance == 0.8
    # 800 accepted over 400 drafts
    assert stats.mean_len == 2.0
    assert stats.accepted == 800
    assert stats.generated == 1000


def test_spec_delta_returns_none_when_nothing_happened():
    c = SpecCounters(draft_tokens=10, accepted=5, drafts=2, per_position=())
    assert spec_delta(c, c) is None


def test_spec_delta_returns_none_on_counter_reset():
    # The server restarted and counters went backwards.
    prev = SpecCounters(draft_tokens=100, accepted=50, drafts=20, per_position=())
    cur = SpecCounters(draft_tokens=10, accepted=5, drafts=2, per_position=())
    assert spec_delta(prev, cur) is None
