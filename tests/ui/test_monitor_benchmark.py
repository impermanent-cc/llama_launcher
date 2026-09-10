from PySide6.QtCore import Qt

from llama_launcher.ui.panels.benchmark_panel import BenchmarkPanel
from llama_launcher.ui.widgets.info_button import InfoButton


def _run(ts="t0", model="qwen.gguf", size=512, extra=None):
    snap = {"model": model}
    if extra:
        snap.update(extra)
    return {
        "timestamp": ts,
        "snapshot": snap,
        "rows": [
            {
                "target_size": size,
                "prompt_n": size + 16,
                "pp_tok_s": 340.0,
                "gen_tok_s": 59.0,
                "total_s": 6.1,
            }
        ],
    }


def _run_with_rows(rows):
    return {"timestamp": "t0", "snapshot": {"model": "m.gguf"}, "rows": rows}


def test_metric_columns_use_fixed_decimals(qtbot):
    """Throughput reads to one decimal and total seconds to two, from the
    full-precision values the run stores, and an integer column passes a
    non-integral stored count through rather than rounding it off."""
    p = BenchmarkPanel()
    qtbot.addWidget(p)
    p.set_benchmark_history(
        [
            _run_with_rows(
                [
                    {
                        "target_size": 2048,
                        "prompt_n": 4,
                        "pp_tok_s": 36.430463408001145,
                        "gen_tok_s": 43.00310070555444,
                        "total_s": 3.0863406666666666,
                    },
                    {"target_size": 512.5},
                ]
            )
        ]
    )
    row = 1  # 0 is the run's group header
    assert p.bench_table.item(row, 0).text() == "2048"
    assert p.bench_table.item(row, 1).text() == "4"
    assert p.bench_table.item(row, 2).text() == "36.4"
    assert p.bench_table.item(row, 3).text() == "43.0"
    assert p.bench_table.item(row, 4).text() == "3.09"
    assert p.bench_table.item(row + 1, 0).text() == "512.5"


def test_metric_columns_right_align(qtbot):
    """Every column of a metric row is right-aligned so the digits line up,
    while the group header row above it keeps its own default alignment."""
    p = BenchmarkPanel()
    qtbot.addWidget(p)
    p.set_benchmark_history([_run()])
    assert not (
        p.bench_table.item(0, 0).textAlignment() & int(Qt.AlignmentFlag.AlignRight)
    )
    for col in range(p.bench_table.columnCount()):
        alignment = p.bench_table.item(1, col).textAlignment()
        assert alignment & int(Qt.AlignmentFlag.AlignRight)


def test_metric_cell_tolerates_non_numeric_and_missing(qtbot):
    """A stale stored value renders as its own text and a missing one as an
    empty cell, rather than raising during the repaint. A stored boolean
    renders as its own text too, rather than as the number 0 or 1."""
    p = BenchmarkPanel()
    qtbot.addWidget(p)
    p.set_benchmark_history(
        [
            _run_with_rows(
                [
                    {
                        "target_size": 128,
                        "prompt_n": None,
                        "pp_tok_s": "n/a",
                        "gen_tok_s": 12,
                        "total_s": None,
                    },
                    {"gen_tok_s": True},
                ]
            )
        ]
    )
    assert p.bench_table.item(1, 1).text() == ""
    assert p.bench_table.item(1, 2).text() == "n/a"
    assert p.bench_table.item(1, 3).text() == "12.0"
    assert p.bench_table.item(1, 4).text() == ""
    assert p.bench_table.item(2, 3).text() == "True"


def test_run_click_emits_config(qtbot):
    p = BenchmarkPanel()
    qtbot.addWidget(p)
    seen = []
    p.benchmark_run_requested.connect(seen.append)
    p.bench_sizes.setText("128, 512")
    p.bench_npredict.setValue(64)
    p.set_benchmark_available(True)
    p.bench_run_btn.click()
    assert seen and seen[0]["sizes"] == [128, 512] and seen[0]["n_predict"] == 64


def test_history_renders_grouped_rows_with_model(qtbot):
    """Each run becomes a labelled group header (model + timestamp) above its
    metric rows, so results stay tied to the model that produced them."""
    p = BenchmarkPanel()
    qtbot.addWidget(p)
    p.set_benchmark_history([_run(ts="t0", model="qwen.gguf", extra={"ngl": "99"})])
    # header row + one metric row
    assert p.bench_table.rowCount() == 2
    header = p.bench_table.item(0, 0).text()
    assert "qwen.gguf" in header and "t0" in header and "-ngl99" in header
    # the metric row carries the size value
    assert p.bench_table.item(1, 0).text() == "512"


def test_history_groups_multiple_runs_newest_first(qtbot):
    p = BenchmarkPanel()
    qtbot.addWidget(p)
    p.set_benchmark_history(
        [
            _run(ts="t1", model="a.gguf", size=128),
            _run(ts="t2", model="b.gguf", size=256),
        ]
    )
    # two headers + two metric rows
    assert p.bench_table.rowCount() == 4
    assert "b.gguf" in p.bench_table.item(0, 0).text()  # newest (t2) first
    assert "a.gguf" in p.bench_table.item(2, 0).text()


def test_empty_history_clears_table(qtbot):
    p = BenchmarkPanel()
    qtbot.addWidget(p)
    p.set_benchmark_history([_run()])
    p.set_benchmark_history([])
    assert p.bench_table.rowCount() == 0


def test_show_run_sets_delta_in_progress(qtbot):
    p = BenchmarkPanel()
    qtbot.addWidget(p)
    delta = {
        "shared": [{"size": 512, "pp_pct": 10.0, "gen_pct": -5.0}],
        "sizes_differ": False,
    }
    p.show_benchmark_run({"rows": []}, delta)
    t = p.bench_progress.text()
    assert "pp +10%" in t and "gen -5%" in t


def test_clear_button_emits_signal(qtbot):
    p = BenchmarkPanel()
    qtbot.addWidget(p)
    with qtbot.waitSignal(p.benchmark_clear_requested, timeout=1000):
        p.bench_clear_btn.click()


def test_table_headers_have_tooltips(qtbot):
    p = BenchmarkPanel()
    qtbot.addWidget(p)
    tips = [
        p.bench_table.horizontalHeaderItem(c).toolTip()
        for c in range(p.bench_table.columnCount())
    ]
    assert all(t.strip() for t in tips)  # every header explained
    assert any("prefill" in t.lower() for t in tips)  # pp t/s explained


def test_legend_explains_metrics(qtbot):
    p = BenchmarkPanel()
    qtbot.addWidget(p)
    assert not hasattr(p, "bench_legend")
    infos = p.findChildren(InfoButton)
    t = " ".join(b.info_text.lower() for b in infos)
    assert "prefill" in t and "generation" in t


def test_available_toggles_run_button(qtbot):
    p = BenchmarkPanel()
    qtbot.addWidget(p)
    p.set_benchmark_available(False)
    assert not p.bench_run_btn.isEnabled()
    p.set_benchmark_available(True)
    assert p.bench_run_btn.isEnabled()
