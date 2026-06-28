from llama_launcher.core.mtp_stats import parse_draft_stats, DraftStats, sparkline

REAL = ("draft acceptance = 0.62008 ( 1797 accepted /  2898 generated), "
        "mean acceptance length =  2.24, acceptance rate per position = (0.727, 0.513)")


def test_parses_real_line():
    d = parse_draft_stats(REAL)
    assert d is not None
    assert abs(d.acceptance - 0.62008) < 1e-6
    assert d.accepted == 1797 and d.generated == 2898
    assert abs(d.mean_len - 2.24) < 1e-6
    assert d.per_position == (0.727, 0.513)


def test_three_positions():
    line = ("draft acceptance = 0.5 ( 3 accepted / 6 generated), "
            "mean acceptance length = 1.5, acceptance rate per position = (0.8, 0.5, 0.2)")
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
    assert sparkline([5.0]) == "▁"


def test_sparkline_flat_series_all_low():
    assert sparkline([5, 5, 5]) == "▁▁▁"


def test_sparkline_ascending_ramp():
    s = sparkline([1, 2, 3, 4, 5, 6, 7, 8])
    assert len(s) == 8
    assert s[0] == "▁" and s[-1] == "█"
    # monotonically non-decreasing block heights
    assert list(s) == sorted(s, key="▁▂▃▄▅▆▇█".index)


def test_sparkline_width_keeps_last_n():
    assert sparkline([1, 2, 3, 4, 5], width=2) == "▁█"
