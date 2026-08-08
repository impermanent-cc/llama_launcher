from llama_launcher.ui.panels.benchmark_panel import BenchmarkPanel


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


def test_show_run_fills_table_and_history(qtbot):
    p = BenchmarkPanel()
    qtbot.addWidget(p)
    run = {"timestamp": "t0", "snapshot": {"ngl": "99", "fa": "on", "ctx": None},
           "rows": [{"target_size": 512, "prompt_n": 528, "pp_tok_s": 340.0,
                     "gen_tok_s": 59.0, "total_s": 6.1}]}
    p.show_benchmark_run(run, None)
    assert p.bench_table.rowCount() == 1
    p.set_benchmark_history([run])
    assert p.bench_history.count() == 1
    assert "-ngl99" in p.bench_history.item(0).text() or "99" in p.bench_history.item(0).text()


def test_available_toggles_run_button(qtbot):
    p = BenchmarkPanel()
    qtbot.addWidget(p)
    p.set_benchmark_available(False)
    assert not p.bench_run_btn.isEnabled()
    p.set_benchmark_available(True)
    assert p.bench_run_btn.isEnabled()
