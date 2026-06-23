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
