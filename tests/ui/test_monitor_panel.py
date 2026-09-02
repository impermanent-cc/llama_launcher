from llama_launcher.core.props import PropsInfo
from llama_launcher.ui.panels.monitor_panel import MonitorPanel
from llama_launcher.ui.widgets.info_button import InfoButton
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


def test_instances_window_has_info_popover(qtbot):
    """The instance cards strip carries an ⓘ describing what the cards show."""
    p = MonitorPanel()
    qtbot.addWidget(p)
    assert isinstance(p.cards_info, InfoButton)
    assert "instance" in p.cards_info.info_text.lower()
    p.cards_info.click()   # popover must not raise


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


def test_update_stats_prefers_live_gen_over_gauge(qtbot):
    # The predicted_tokens_seconds gauge reads 0 mid-generation; the live
    # n_decode_total rate must win so the user sees real throughput, not 0.
    p = MonitorPanel()
    qtbot.addWidget(p)
    p.update_stats({"tok_s": 0.0, "gen_tok_s_live": 30.0, "prompt_tok_s": None,
                    "kv_pct": None, "gpus": [], "cpu": "", "mem": "", "uptime": "",
                    "metrics_on": True})
    assert "gen 30.0" in p._summary_text()
    assert "gen 0.0" not in p._summary_text()


def test_update_stats_falls_back_to_gauge_when_no_live(qtbot):
    # Idle between requests: no live rate, so show the last request's gauge value.
    p = MonitorPanel()
    qtbot.addWidget(p)
    p.update_stats({"tok_s": 25.0, "gen_tok_s_live": None, "prompt_tok_s": None,
                    "kv_pct": None, "gpus": [], "cpu": "", "mem": "", "uptime": "",
                    "metrics_on": True})
    assert "gen 25.0" in p._summary_text()


def test_throughput_bar_uses_live_gen(qtbot):
    # The sparkline/number must track the live rate, not the frozen 0 gauge.
    p = MonitorPanel()
    qtbot.addWidget(p)
    base = {"tok_s": 0.0, "prompt_tok_s": None, "kv_pct": None, "gpus": [],
            "cpu": "", "mem": "", "uptime": "", "metrics_on": True}
    p.update_stats({**base, "gen_tok_s_live": 40.0})
    p.update_stats({**base, "gen_tok_s_live": 42.0})
    assert not p.throughput_label.isHidden()
    assert "42" in p.throughput_label.text()


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


def _rows():
    return [
        {"name": "llama-a", "profile": "a", "port": 8080, "running": True,
         "health": "ready", "stat": "10 tok/s", "tok_s": 10.0, "kv_pct": 0.2,
         "embeddings": False, "reranking": False, "mode": "server"},
        {"name": "llama-b", "profile": "b", "port": 8081, "running": True,
         "health": "ready", "stat": "ready", "tok_s": None, "kv_pct": None,
         "embeddings": True, "reranking": False, "mode": "server"},
    ]


def test_set_instance_cards_builds_and_selects(qtbot):
    panel = MonitorPanel(); qtbot.addWidget(panel)
    panel.set_instance_cards({"rows": _rows(), "selected_name": "llama-b"})
    assert panel.card_names() == ["llama-a", "llama-b"]
    assert panel.selected_card_name() == "llama-b"
    assert "10 tok/s" in panel.card("llama-a").headline_text()
    assert panel.card("llama-b").headline_text() == "ready"


def test_set_instance_cards_updates_in_place_when_membership_stable(qtbot):
    panel = MonitorPanel(); qtbot.addWidget(panel)
    panel.set_instance_cards({"rows": _rows(), "selected_name": None})
    first = panel.card("llama-a")
    panel.set_instance_cards({"rows": _rows(), "selected_name": None})  # same names
    assert panel.card("llama-a") is first          # widget reused, not rebuilt


def test_set_instance_cards_rebuilds_on_membership_change(qtbot):
    panel = MonitorPanel(); qtbot.addWidget(panel)
    panel.set_instance_cards({"rows": _rows(), "selected_name": None})
    panel.set_instance_cards({"rows": _rows()[:1], "selected_name": None})  # llama-b gone
    assert panel.card_names() == ["llama-a"]


def test_card_stop_button_emits_instance_stop_requested(qtbot):
    panel = MonitorPanel(); qtbot.addWidget(panel)
    panel.set_instance_cards({"rows": _rows(), "selected_name": None})
    got = []; panel.instance_stop_requested.connect(got.append)
    panel.card("llama-a").stop_button().click()
    assert got == ["llama-a"]


def test_card_click_emits_instance_selected(qtbot):
    panel = MonitorPanel(); qtbot.addWidget(panel)
    panel.set_instance_cards({"rows": _rows(), "selected_name": None})
    got = []; panel.instance_selected.connect(got.append)
    panel.card("llama-a").selected.emit("llama-a")   # simulate a card click
    assert got == ["llama-a"]


def test_monitor_panel_has_no_benchmark_widgets(qtbot):
    # Benchmark lives in its own BenchmarkPanel/tab; the Monitor panel is
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
    assert not hasattr(panel, "bench_legend")
    # legend text lives behind an on-demand InfoButton popover
    assert panel.findChildren(InfoButton)
    # no separate history list; runs live in the grouped table
    assert not hasattr(panel, "bench_history")


def test_card_title_rpc_worker_includes_node(qtbot):
    """MonitorPanel._card_title special-cases an rpc-worker Instance, which
    otherwise shares its pool head's profile name/port (worker containers
    carry the SAME `llama-launcher.profile` label), so its StatCard title
    identifies the worker's own node instead of duplicating the head's."""
    from llama_launcher.core.instances import Instance
    p = MonitorPanel(); qtbot.addWidget(p)
    inst = Instance(name="llama-pool-rpc1", profile="pool", mode="rpc-worker",
                    running=True, port=None, host="127.0.0.1",
                    embeddings=False, reranking=False, node="box2")
    label = p._card_title(inst)
    assert "rpc-worker" in label and "box2" in label


def test_card_title_non_worker_instance_uses_profile_name(qtbot):
    from llama_launcher.core.instances import Instance
    p = MonitorPanel(); qtbot.addWidget(p)
    inst = Instance(name="llama-a", profile="a", mode="server", running=True,
                    port=8080, host="127.0.0.1", embeddings=False, reranking=False)
    assert p._card_title(inst) == "a"


def test_monitor_stats_legend_explains_gen_prompt_kv(qtbot):
    p = MonitorPanel()
    qtbot.addWidget(p)
    assert not hasattr(p, "stats_legend")
    # legend text lives ONLY on the compact InfoButton (popover + its own
    # hover), NOT on the full-width summary bar: a tooltip there fires on
    # hover anywhere along the bar, duplicating the button.
    assert p.summary.toolTip() == ""
    btn = p.findChild(InfoButton)
    assert btn is not None
    t = btn.info_text.lower()
    assert "gen" in t and "prompt" in t and "kv" in t
    assert "idle" in t          # explains why they can read 0 while running


def test_update_stats_prefers_live_prompt_rate(qtbot):
    """Mid-prefill the completion gauge holds the LAST request's rate; the live
    slot-delta rate is the current one and must win when present."""
    p = MonitorPanel()
    qtbot.addWidget(p)
    p.update_stats({"metrics_on": True, "prompt_tok_s": 12.0,
                    "prompt_tok_s_live": 812.0})
    assert "prompt 812" in p.summary.text()
    assert "prompt 12 " not in p.summary.text()


def test_update_stats_prompt_falls_back_to_gauge(qtbot):
    p = MonitorPanel()
    qtbot.addWidget(p)
    p.update_stats({"metrics_on": True, "prompt_tok_s": 12.0,
                    "prompt_tok_s_live": None})
    assert "prompt 12" in p.summary.text()
