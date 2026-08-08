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


class _StubWorker:
    """Stand-in for a running _UpdateWorker/QThread — no real thread involved."""

    def __init__(self, wait_returns):
        self._wait_returns = wait_returns
        self.wait_calls = 0
        self.terminated = False

    def isRunning(self):
        return True

    def wait(self, ms):
        self.wait_calls += 1
        if callable(self._wait_returns):
            return self._wait_returns(ms)
        return self._wait_returns

    def terminate(self):
        self.terminated = True


def _ensure_no_benchmark(win):
    # A fresh MainWindow never started a benchmark; make sure of it so
    # _stop_timers's benchmark block (which runs after the fetch/update
    # drain) early-returns cleanly instead of touching our stub.
    assert getattr(win, "_benchmark_thread", None) is None


def test_stop_timers_drains_fetch_worker_that_finishes_promptly(win):
    _ensure_no_benchmark(win)
    stub = _StubWorker(wait_returns=True)
    win._fetch_worker = stub
    win._stop_timers()          # must not raise/hang
    assert stub.wait_calls >= 1
    assert stub.terminated is False


def test_stop_timers_terminates_fetch_worker_that_never_finishes(win):
    _ensure_no_benchmark(win)
    stub = _StubWorker(wait_returns=False)
    win._fetch_worker = stub
    win._stop_timers()          # must return, not hang (100-iteration ceiling)
    # 100 ceiling attempts, plus the post-terminate() backstop wait().
    assert stub.wait_calls >= 100
    assert stub.terminated is True


def test_update_badge_disabled_during_fetch_and_reenabled_on_finish(win, monkeypatch):
    win.image_edit.setText("ghcr.io/ggml-org/llama.cpp:server-cuda12")
    monkeypatch.setattr("llama_launcher.ui.main_window._UpdateWorker.start",
                        lambda self: None)
    # No found/failed fires in this test, so _on_fetch_finished's "no newer
    # build" path would pop a real modal dialog and block the test; stub it
    # out like the existing finished-state tests do.
    monkeypatch.setattr("llama_launcher.ui.main_window.QMessageBox.information",
                        lambda *a, **k: None)
    win.on_fetch_latest()
    assert win.update_badge.isEnabled() is False
    win._on_fetch_finished()
    assert win.update_badge.isEnabled() is True
