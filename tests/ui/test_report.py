import llama_launcher.ui.main_window as mw
from llama_launcher.ui.dialogs.report_dialog import ReportDialog
from llama_launcher.core.report import REPORT_SECTIONS
from llama_launcher.core.spec import Profile, Mount, Runtime


def test_report_dialog_sections(qtbot):
    d = ReportDialog(initial={s: (s != "logs") for s in REPORT_SECTIONS})
    qtbot.addWidget(d)
    sel = d.selected_sections()
    assert set(sel.keys()) == set(REPORT_SECTIONS)
    assert sel["logs"] is False and sel["command"] is True


def test_gather_report_data_redacts(qtbot, monkeypatch):
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda: [])
    monkeypatch.setattr(mw.runtime, "is_rootless", lambda b: False)
    w = mw.MainWindow(); qtbot.addWidget(w)
    w.load_profile(Profile(name="r", image="img", runtime=Runtime(binary="podman"),
                           mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
                           model="/models/m.gguf",
                           settings={"port": 8080, "api-key": "SEKRET"}))
    data = w.gather_report_data()
    assert "SEKRET" not in data["command"]      # redacted in the command string
    assert "image" in data and "runtime" in data


def test_gather_report_data_redacts_logs(qtbot, monkeypatch):
    """Secrets in monitor logs must be redacted by gather_report_data()."""
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda: [])
    monkeypatch.setattr(mw.runtime, "is_rootless", lambda b: False)
    w = mw.MainWindow(); qtbot.addWidget(w)
    w.load_profile(Profile(name="rl", image="img", runtime=Runtime(binary="podman"),
                           mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
                           model="/models/m.gguf",
                           settings={"port": 8080}))
    # Inject a secret into the log view
    w.monitor_panel.log_view.setPlainText("server started\n--api-key SEKRET\nAuthorization: Bearer BEARTOKEN")
    data = w.gather_report_data()
    assert "SEKRET" not in data["logs"], "api-key secret leaked from logs into report data"
    assert "BEARTOKEN" not in data["logs"], "bearer token leaked from logs into report data"
