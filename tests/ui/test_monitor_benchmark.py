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
