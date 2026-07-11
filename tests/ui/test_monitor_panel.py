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


def test_reset_clears_endpoints(qtbot):
    p = MonitorPanel()
    qtbot.addWidget(p)
    p.set_endpoints(8080, embeddings=True, reranking=False)
    p.reset()
    assert p.endpoints_label.isHidden()
