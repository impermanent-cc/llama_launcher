from llama_launcher.ui.main_window import MainWindow
import llama_launcher.ui.main_window as mw

_DRAFT_LINE = ("draft acceptance = 0.62008 ( 1797 accepted /  2898 generated), "
               "mean acceptance length =  2.24, acceptance rate per position = (0.727, 0.513)")


def test_on_launch_resets_monitor(qtbot, monkeypatch):
    w = MainWindow()
    qtbot.addWidget(w)
    monkeypatch.setattr(w._launch, "_validate_or_warn", lambda: True)
    monkeypatch.setattr(w._launch, "vram_check", lambda: None)
    monkeypatch.setattr(mw.terminal, "launch", lambda argv: None)
    # seed monitor state, then launch
    w.monitor_panel.append_log(_DRAFT_LINE + "\n")
    assert not w.monitor_panel.mtp_label.isHidden()
    w._launch.on_launch()
    assert w.monitor_panel.mtp_label.isHidden()
    assert w.monitor_panel.log_view.toPlainText() == ""


def test_on_launch_sets_endpoints(qtbot, monkeypatch):
    w = MainWindow()
    qtbot.addWidget(w)
    monkeypatch.setattr(w._launch, "_validate_or_warn", lambda: True)
    monkeypatch.setattr(w._launch, "vram_check", lambda: None)
    monkeypatch.setattr(mw.terminal, "launch", lambda argv: None)
    w._configure_panel._widgets["embeddings"].set_value(True)
    w._configure_panel._widgets["port"].set_value(8080)
    w._launch.on_launch()
    assert not w.monitor_panel.endpoints_label.isHidden()
    assert "/v1/embeddings" in w.monitor_panel.endpoints_label.text()
    assert "8080" in w.monitor_panel.endpoints_label.text()
