import pytest

from llama_launcher.ui.main_window import MainWindow


@pytest.fixture
def win(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def test_fetch_button_has_tooltip(win):
    assert win.fetch_btn.toolTip().strip()
    # the tooltip must note the not-downloaded caveat
    assert "pull" in win.fetch_btn.toolTip().lower()


def test_fetch_button_sits_next_to_detect(win):
    # Both act on the Image field, so they share the Image row's parent widget.
    assert win.fetch_btn.parentWidget() is win.detect_image_btn.parentWidget()


def test_fetch_button_relabeled(win):
    assert win.fetch_btn.text() == "Fetch latest"


def test_no_image_shows_message_and_starts_no_worker(win, monkeypatch):
    win.image_edit.setText("")
    seen = {}
    monkeypatch.setattr("llama_launcher.ui.main_window.QMessageBox.information",
                        lambda *a, **k: seen.setdefault("info", True))
    # If a worker were started, start() would run; stub it to detect that.
    started = {}
    monkeypatch.setattr("llama_launcher.ui.main_window._UpdateWorker.start",
                        lambda self: started.setdefault("started", True))
    win.on_fetch_latest()
    assert seen.get("info") and not started.get("started")


def test_click_enters_working_state_and_starts_worker(win, monkeypatch):
    win.image_edit.setText("ghcr.io/ggml-org/llama.cpp:server-cuda12")
    monkeypatch.setattr("llama_launcher.ui.main_window._UpdateWorker.start",
                        lambda self: None)          # don't spin a real thread
    win.on_fetch_latest()
    assert win.fetch_btn.text() == "Fetching…"
    assert win.fetch_btn.isEnabled() is False
    assert win._fetch_worker is not None
    assert win._fetch_repo == "ghcr.io/ggml-org/llama.cpp"


def test_found_sets_tag_and_notes_not_downloaded(win, monkeypatch):
    win.image_edit.setText("ghcr.io/ggml-org/llama.cpp:server-cuda12")
    monkeypatch.setattr("llama_launcher.ui.main_window._UpdateWorker.start",
                        lambda self: None)
    info = {}
    monkeypatch.setattr("llama_launcher.ui.main_window.QMessageBox.information",
                        lambda *a, **k: info.setdefault("msg", a[2] if len(a) > 2 else ""))
    win.on_fetch_latest()
    win._on_fetch_found("server-cuda12-b9999")
    assert win.image_edit.text() == "ghcr.io/ggml-org/llama.cpp:server-cuda12-b9999"
    assert "pull" in str(info.get("msg", "")).lower()   # "not downloaded, pull ..."


def test_failed_shows_warning(win, monkeypatch):
    win.image_edit.setText("ghcr.io/ggml-org/llama.cpp:server-cuda12")
    monkeypatch.setattr("llama_launcher.ui.main_window._UpdateWorker.start",
                        lambda self: None)
    warned = {}
    monkeypatch.setattr("llama_launcher.ui.main_window.QMessageBox.warning",
                        lambda *a, **k: warned.setdefault("w", True))
    win.on_fetch_latest()
    win._on_fetch_failed("connection refused")
    assert warned.get("w")


def test_finished_restores_button_and_reports_no_newer(win, monkeypatch):
    win.image_edit.setText("ghcr.io/ggml-org/llama.cpp:server-cuda12")
    monkeypatch.setattr("llama_launcher.ui.main_window._UpdateWorker.start",
                        lambda self: None)
    info = {}
    monkeypatch.setattr("llama_launcher.ui.main_window.QMessageBox.information",
                        lambda *a, **k: info.setdefault("msg", True))
    win.on_fetch_latest()          # sets _fetch_got_result = False
    win._on_fetch_finished()       # no found/failed fired -> "no newer build"
    assert win.fetch_btn.isEnabled() and win.fetch_btn.text() == "Fetch latest"
    assert info.get("msg")


def test_finished_after_result_does_not_report_no_newer(win, monkeypatch):
    win.image_edit.setText("ghcr.io/ggml-org/llama.cpp:server-cuda12")
    monkeypatch.setattr("llama_launcher.ui.main_window._UpdateWorker.start",
                        lambda self: None)
    monkeypatch.setattr("llama_launcher.ui.main_window.QMessageBox.information",
                        lambda *a, **k: None)
    win.on_fetch_latest()
    win._on_fetch_found("server-cuda12-b9999")   # a result arrived
    seen = {}
    monkeypatch.setattr("llama_launcher.ui.main_window.QMessageBox.information",
                        lambda *a, **k: seen.setdefault("msg", True))
    win._on_fetch_finished()
    assert win.fetch_btn.isEnabled() and win.fetch_btn.text() == "Fetch latest"
    assert not seen.get("msg")     # no "no newer build" message when a result came
