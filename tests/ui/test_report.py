import llama_launcher.ui.main_window as mw
from llama_launcher.core.report import REPORT_SECTIONS
from llama_launcher.core.spec import Mount, Profile, Runtime
from llama_launcher.ui.controllers import report_controller
from llama_launcher.ui.dialogs.report_dialog import ReportDialog


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
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(
        Profile(
            name="r",
            image="img",
            runtime=Runtime(binary="podman"),
            mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
            model="/models/m.gguf",
            settings={"port": 8080, "api-key": "SEKRET"},
        )
    )
    data = w._report.gather_report_data()
    assert "SEKRET" not in data["command"]  # redacted in the command string
    assert "image" in data and "runtime" in data


def test_gather_report_data_redacts_logs(qtbot, monkeypatch):
    """Secrets in monitor logs must be redacted by gather_report_data()."""
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda: [])
    monkeypatch.setattr(mw.runtime, "is_rootless", lambda b: False)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(
        Profile(
            name="rl",
            image="img",
            runtime=Runtime(binary="podman"),
            mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
            model="/models/m.gguf",
            settings={"port": 8080},
        )
    )
    # Inject a secret into the log view
    w.monitor_panel.log_view.setPlainText(
        "server started\n--api-key SEKRET\nAuthorization: Bearer BEARTOKEN"
    )
    data = w._report.gather_report_data()
    assert "SEKRET" not in data["logs"], (
        "api-key secret leaked from logs into report data"
    )
    assert "BEARTOKEN" not in data["logs"], (
        "bearer token leaked from logs into report data"
    )


def test_gather_report_data_includes_metrics(qtbot, monkeypatch):
    """When --metrics is enabled and the server responds, gather_report_data()
    captures throughput and KV-cache usage in the report's metrics section."""
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda: [])
    monkeypatch.setattr(mw.runtime, "is_rootless", lambda b: False)
    monkeypatch.setattr(
        mw.metrics,
        "fetch_metrics",
        lambda port, timeout=1.0, **kw: {
            "llamacpp:predicted_tokens_seconds": 42.0,
            "llamacpp:prompt_tokens_seconds": 800.0,
        },
    )
    monkeypatch.setattr(
        mw.metrics,
        "fetch_slots",
        lambda port, timeout=1.0, **kw: [
            {"n_ctx": 100, "n_prompt_tokens_processed": 25},
        ],
    )
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(
        Profile(
            name="m",
            image="img",
            runtime=Runtime(binary="podman"),
            mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
            model="/models/m.gguf",
            settings={"port": 8080, "metrics": True},
        )
    )
    data = w._report.gather_report_data()
    assert "metrics" in data
    assert "42" in data["metrics"]  # generation tok/s
    assert "25%" in data["metrics"]  # KV usage = 25 / 100


def test_gather_report_data_metrics_off_note(qtbot, monkeypatch):
    """With --metrics off, the report explains how to enable it and makes no
    network call to /metrics."""
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda: [])
    monkeypatch.setattr(mw.runtime, "is_rootless", lambda b: False)
    hits = {"n": 0}
    monkeypatch.setattr(
        mw.metrics,
        "fetch_metrics",
        lambda *a, **k: hits.__setitem__("n", hits["n"] + 1) or {},
    )
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(
        Profile(
            name="m",
            image="img",
            runtime=Runtime(binary="podman"),
            mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
            model="/models/m.gguf",
            settings={"port": 8080},
        )
    )
    data = w._report.gather_report_data()
    assert "metrics" in data
    assert "--metrics" in data["metrics"]
    assert hits["n"] == 0


def test_save_report_writes_file_and_contains_timestamp(qtbot, tmp_path, monkeypatch):
    """_save_report() writes a file under <base_dir>/reports/ with the right name and
    the markdown contains the _Generated: header."""
    monkeypatch.setattr(mw, "base_dir", lambda: tmp_path)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    md = "# Llama Launcher diagnostic report\n\n_Generated: 20260101-120000_\n\nsome content"
    saved = w._report._save_report(md)
    # File exists under reports/ dir
    matches = list((tmp_path / "reports").glob("llama-launcher-report-*.md"))
    assert len(matches) == 1
    assert matches[0] == saved
    assert "_Generated:" in matches[0].read_text()


def test_on_generate_report_auto_saves(qtbot, tmp_path, monkeypatch):
    """on_generate_report() auto-saves to reports/ without requiring QFileDialog."""
    monkeypatch.setattr(mw, "base_dir", lambda: tmp_path)
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    monkeypatch.setattr(mw.gpu, "query_gpus", lambda: [])
    monkeypatch.setattr(mw.runtime, "is_rootless", lambda b: False)
    # Stub dialogs to avoid blocking
    monkeypatch.setattr(
        report_controller.QMessageBox, "information", lambda *a, **k: None
    )
    # Stub ReportDialog to always return accepted with all sections enabled
    from llama_launcher.core.report import REPORT_SECTIONS

    class _FakeDialog:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            return True

        def selected_sections(self):
            return {s: True for s in REPORT_SECTIONS}

    monkeypatch.setattr(
        "llama_launcher.ui.controllers.report_controller.ReportDialog", _FakeDialog
    )
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(
        Profile(
            name="rpt",
            image="img",
            runtime=Runtime(binary="podman"),
            mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
            model="/models/m.gguf",
            settings={"port": 8080},
        )
    )
    w._report.on_generate_report()
    matches = list((tmp_path / "reports").glob("llama-launcher-report-*.md"))
    assert len(matches) == 1
    content = matches[0].read_text()
    assert "_Generated:" in content


def test_export_sh_uses_shlex_join_and_owner_only_mode(qtbot, monkeypatch, tmp_path):
    """Export must shell-quote each argv element (a profile field with a space or
    metacharacter must not become shell code) and write an owner-only file."""
    import stat

    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    monkeypatch.setattr(
        w._configure_panel,
        "build_current_command",
        lambda *a: ["podman", "run", "--name", "a b; touch pwned", "img"],
    )
    out = tmp_path / "run.sh"
    w._report.export_sh(str(out))
    text = out.read_text()
    assert "'a b; touch pwned'" in text  # single argv token, quoted
    assert "touch pwned\n" not in text  # not split into a second command
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
