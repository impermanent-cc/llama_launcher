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
