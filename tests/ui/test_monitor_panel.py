from llama_launcher.core.props import PropsInfo
from llama_launcher.ui.panels.monitor_panel import MonitorPanel
from llama_launcher.services.gpu import GpuStat


def test_monitor_renders_stats(qtbot):
    p = MonitorPanel()
    qtbot.addWidget(p)
    p.update_stats({
        "tok_s": 42.5, "prompt_tok_s": 300.0, "kv_pct": 0.25,
        "gpus": [GpuStat("RTX 4090", 8192, 24576, 16384, 37, 55)],
        "cpu": "12.5%", "mem": "1.2GB / 16GB", "uptime": "00:01:23",
        "metrics_on": True,
    })
    text = p._summary_text()        # helper that returns the rendered summary string
    assert "42.5" in text and "25" in text and "RTX 4090" in text


def test_monitor_prompts_to_enable_metrics(qtbot):
    p = MonitorPanel()
    qtbot.addWidget(p)
    p.update_stats({"tok_s": None, "prompt_tok_s": None, "kv_pct": None,
                    "gpus": [], "cpu": "", "mem": "", "uptime": "", "metrics_on": False})
    assert "enable --metrics" in p._summary_text().lower()


def test_append_log(qtbot):
    p = MonitorPanel()
    qtbot.addWidget(p)
    p.append_log("loaded model\n")
    assert "loaded model" in p.log_view.toPlainText()


def test_enable_metrics_button_shown_when_metrics_off(qtbot):
    """'Enable --metrics & relaunch' button is shown (not hidden) when metrics_on is False."""
    p = MonitorPanel()
    qtbot.addWidget(p)
    p.update_stats({"tok_s": None, "prompt_tok_s": None, "kv_pct": None,
                    "gpus": [], "cpu": "", "mem": "", "uptime": "", "metrics_on": False})
    assert not p.enable_metrics_btn.isHidden()


def test_enable_metrics_button_hidden_when_metrics_on(qtbot):
    """'Enable --metrics & relaunch' button is hidden when metrics_on is True."""
    p = MonitorPanel()
    qtbot.addWidget(p)
    p.update_stats({"tok_s": 1.0, "prompt_tok_s": None, "kv_pct": None,
                    "gpus": [], "cpu": "", "mem": "", "uptime": "", "metrics_on": True})
    assert p.enable_metrics_btn.isHidden()


def test_enable_metrics_button_emits_signal(qtbot):
    """Clicking the button emits enable_metrics_requested signal."""
    p = MonitorPanel()
    qtbot.addWidget(p)
    p.update_stats({"tok_s": None, "prompt_tok_s": None, "kv_pct": None,
                    "gpus": [], "cpu": "", "mem": "", "uptime": "", "metrics_on": False})
    with qtbot.waitSignal(p.enable_metrics_requested, timeout=1000):
        p.enable_metrics_btn.click()


_DRAFT_LINE = ("draft acceptance = 0.62008 ( 1797 accepted /  2898 generated), "
               "mean acceptance length =  2.24, acceptance rate per position = (0.727, 0.513)")


def test_mtp_label_appears_from_log(qtbot):
    p = MonitorPanel()
    qtbot.addWidget(p)
    assert p.mtp_label.isHidden()
    p.append_log(_DRAFT_LINE + "\n")
    assert not p.mtp_label.isHidden()
    t = p.mtp_label.text()
    assert "62%" in t and "2.24" in t and "73%" in t


def test_mtp_line_split_across_chunks(qtbot):
    p = MonitorPanel()
    qtbot.addWidget(p)
    half = len(_DRAFT_LINE) // 2
    p.append_log(_DRAFT_LINE[:half])           # no newline yet
    assert p.mtp_label.isHidden()
    p.append_log(_DRAFT_LINE[half:] + "\n")    # completes the line
    assert not p.mtp_label.isHidden()
    assert "62%" in p.mtp_label.text()


def test_reset_clears_mtp_and_logs(qtbot):
    p = MonitorPanel()
    qtbot.addWidget(p)
    p.append_log(_DRAFT_LINE + "\n")
    p.reset()
    assert p.mtp_label.isHidden()
    assert p.log_view.toPlainText() == ""


def _stats(tok, metrics_on=True):
    return {"tok_s": tok, "prompt_tok_s": None, "kv_pct": None,
            "gpus": [], "cpu": "", "mem": "", "uptime": "", "metrics_on": metrics_on}


def test_throughput_sparkline_appears(qtbot):
    p = MonitorPanel()
    qtbot.addWidget(p)
    p.update_stats(_stats(40.0))
    p.update_stats(_stats(60.0))
    assert not p.throughput_label.isHidden()
    assert "tok/s" in p.throughput_label.text()


def test_throughput_hidden_without_metrics(qtbot):
    p = MonitorPanel()
    qtbot.addWidget(p)
    p.update_stats(_stats(None, metrics_on=False))
    assert p.throughput_label.isHidden()


def test_reset_clears_throughput_history(qtbot):
    p = MonitorPanel()
    qtbot.addWidget(p)
    p.update_stats(_stats(40.0))
    p.reset()
    assert p.throughput_label.isHidden()
    assert len(p._tok_history) == 0


def test_set_endpoints_shows_embedding_url(qtbot):
    p = MonitorPanel()
    qtbot.addWidget(p)
    p.set_endpoints(8080, embeddings=True, reranking=False)
    assert not p.endpoints_label.isHidden()
    assert "/v1/embeddings" in p.endpoints_label.text()
    assert "8080" in p.endpoints_label.text()


def test_set_endpoints_shows_rerank_url(qtbot):
    p = MonitorPanel()
    qtbot.addWidget(p)
    p.set_endpoints(9000, embeddings=True, reranking=True)
    assert "/v1/rerank" in p.endpoints_label.text()


def test_set_endpoints_hidden_when_neither(qtbot):
    p = MonitorPanel()
    qtbot.addWidget(p)
    p.set_endpoints(8080, embeddings=False, reranking=False)
    assert p.endpoints_label.isHidden()


def test_set_endpoints_rerank_only(qtbot):
    p = MonitorPanel()
    qtbot.addWidget(p)
    p.set_endpoints(8080, embeddings=False, reranking=True)
    assert not p.endpoints_label.isHidden()
    assert "/v1/rerank" in p.endpoints_label.text()
    assert "/v1/embeddings" not in p.endpoints_label.text()


def test_reset_clears_endpoints(qtbot):
    p = MonitorPanel()
    qtbot.addWidget(p)
    p.set_endpoints(8080, embeddings=True, reranking=False)
    p.reset()
    assert p.endpoints_label.isHidden()


def test_set_props_renders_all_fields(qtbot):
    panel = MonitorPanel()
    qtbot.addWidget(panel)
    panel.set_props(PropsInfo(build="b9755-0ef6f06d5", n_ctx=8192,
                              model_alias="qwen35moe", total_slots=2,
                              modalities={"vision": True, "audio": False}))
    assert not panel.info_label.isHidden()
    t = panel.info_label.text()
    assert "b9755-0ef6f06d5" in t and "8192" in t and "qwen35moe" in t
    assert "vision" in t and "2 slots" in t


def test_set_props_none_hides_label(qtbot):
    panel = MonitorPanel()
    qtbot.addWidget(panel)
    panel.set_props(PropsInfo("b", 1, "a", 1, {}))
    panel.set_props(None)
    assert panel.info_label.isHidden()


def test_set_props_omits_absent_fields(qtbot):
    panel = MonitorPanel()
    qtbot.addWidget(panel)
    panel.set_props(PropsInfo(build=None, n_ctx=4096, model_alias=None,
                              total_slots=None, modalities={}))
    t = panel.info_label.text()
    assert "4096" in t and "build" not in t and "slots" not in t


def test_reset_hides_info_label(qtbot):
    panel = MonitorPanel()
    qtbot.addWidget(panel)
    panel.set_props(PropsInfo("b", 1, "a", 1, {}))
    panel.reset()
    assert panel.info_label.isHidden()


def test_update_stats_shows_speculating_indicator(qtbot):
    panel = MonitorPanel()
    qtbot.addWidget(panel)
    panel.update_stats({"metrics_on": False, "speculating": True})
    assert "spec ●" in panel.summary.text()


def test_update_stats_omits_indicator_when_not_speculating(qtbot):
    panel = MonitorPanel()
    qtbot.addWidget(panel)
    panel.update_stats({"metrics_on": False, "speculating": False})
    assert "spec ●" not in panel.summary.text()


def test_set_instances_renders_rows_and_selects(qtbot):
    from llama_launcher.ui.panels.monitor_panel import MonitorPanel
    panel = MonitorPanel(); qtbot.addWidget(panel)
    rows = [
        {"name": "llama-a", "profile": "a", "port": 8080, "running": True,
         "health": "ready", "stat": "64 tok/s"},
        {"name": "llama-b", "profile": "b", "port": 8081, "running": True,
         "health": "ready", "stat": "ready"},
    ]
    panel.set_instances(rows, selected_name="llama-b")
    assert panel.instances_table.rowCount() == 2
    assert panel.selected_instance_name() == "llama-b"


def test_instance_row_click_emits_selected(qtbot):
    from llama_launcher.ui.panels.monitor_panel import MonitorPanel
    panel = MonitorPanel(); qtbot.addWidget(panel)
    panel.set_instances([{"name": "llama-a", "profile": "a", "port": 8080,
                          "running": True, "health": "ready", "stat": ""}], None)
    got = []
    panel.instance_selected.connect(got.append)
    panel._emit_selected_for_row(0)            # what a row click calls
    assert got == ["llama-a"]


def test_instance_stop_button_emits(qtbot):
    from llama_launcher.ui.panels.monitor_panel import MonitorPanel
    panel = MonitorPanel(); qtbot.addWidget(panel)
    panel.set_instances([{"name": "llama-a", "profile": "a", "port": 8080,
                          "running": True, "health": "ready", "stat": ""}], None)
    got = []
    panel.instance_stop_requested.connect(got.append)
    btn = panel.instances_table.cellWidget(0, 4)   # column 4 is the Stop button
    btn.click()
    assert got == ["llama-a"]


def test_monitor_panel_has_no_benchmark_widgets(qtbot):
    # Benchmark moved to its own BenchmarkPanel/tab; the Monitor panel is now
    # just instances + logs and must not carry the benchmark controls.
    from llama_launcher.ui.panels.monitor_panel import MonitorPanel
    panel = MonitorPanel(); qtbot.addWidget(panel)
    for attr in ("bench_table", "bench_history", "bench_run_btn", "show_benchmark_run"):
        assert not hasattr(panel, attr), attr


def test_benchmark_panel_has_controls_table_and_clear(qtbot):
    from llama_launcher.ui.panels.benchmark_panel import BenchmarkPanel
    panel = BenchmarkPanel(); qtbot.addWidget(panel)
    assert panel.bench_run_btn is not None
    assert panel.bench_table is not None
    assert panel.bench_clear_btn is not None
    assert panel.bench_legend is not None
    # the separate history list was folded into the grouped table
    assert not hasattr(panel, "bench_history")


def test_monitor_stats_legend_explains_gen_prompt_kv(qtbot):
    p = MonitorPanel()
    qtbot.addWidget(p)
    t = p.stats_legend.text().lower()
    assert "gen" in t and "prompt" in t and "kv" in t
    assert "idle" in t          # explains why they can read 0 while running
