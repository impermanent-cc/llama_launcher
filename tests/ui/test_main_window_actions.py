import pytest

import llama_launcher.ui.main_window as mw
from llama_launcher.core.spec import Profile, Mount, Runtime


@pytest.fixture(autouse=True)
def _stub_dialogs(monkeypatch):
    # Modal dialogs would block forever in the headless/offscreen test run.
    monkeypatch.setattr(mw.QMessageBox, "critical", lambda *a, **k: None)
    monkeypatch.setattr(mw.QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(mw.QMessageBox, "information", lambda *a, **k: None)


def _profile():
    return Profile(name="Act", image="img:tag",
                   runtime=Runtime(binary="podman", gpu_mode="cdi"),
                   mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
                   model="/models/m.gguf", settings={"port": 8080})


def test_save_and_reload_profile(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(mw, "base_dir", lambda: tmp_path)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w.load_profile(_profile())
    w.save_current_profile()
    # new window sees the saved profile in its dropdown
    w2 = mw.MainWindow()
    qtbot.addWidget(w2)
    names = [w2.profile_combo.itemText(i) for i in range(w2.profile_combo.count())]
    assert "Act" in names


def test_name_field_is_saved(qtbot, tmp_path, monkeypatch):
    """A name typed into the Name field is the name the profile saves under."""
    monkeypatch.setattr(mw, "base_dir", lambda: tmp_path)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w.load_profile(_profile())            # name="Act"
    w.name_edit.setText("Renamed Build")
    w.save_current_profile()
    w2 = mw.MainWindow()
    qtbot.addWidget(w2)
    names = [w2.profile_combo.itemText(i) for i in range(w2.profile_combo.count())]
    assert "Renamed Build" in names


def test_on_launch_does_not_follow_logs_before_container_exists(qtbot, monkeypatch):
    """The container is created asynchronously by the launched terminal, so right
    after on_launch() it does not exist yet. Attaching `podman logs -f` then just
    captures 'no such container' and dies, leaving the logs pane stuck on it. So
    on_launch must NOT start a follower against a not-yet-existing container."""
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    monkeypatch.setattr(mw.terminal, "launch", lambda *a, **k: None)
    monkeypatch.setattr(mw.runtime, "container_exists", lambda name, binary: False)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w.load_profile(_profile())
    w.on_launch()
    assert w._log_proc is None


def test_start_log_follower_skips_when_container_absent(qtbot, monkeypatch):
    monkeypatch.setattr(mw.runtime, "container_exists", lambda name, binary: False)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._start_log_follower()
    assert w._log_proc is None


def test_update_status_starts_follower_when_running(qtbot, monkeypatch):
    """Once the container is actually running and no follower is attached, the
    status poll starts one (logs replay from the start, so nothing is missed)."""
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    monkeypatch.setattr(mw.runtime, "container_state", lambda name, binary: "running")
    monkeypatch.setattr(mw.health, "health_ok", lambda port: True)
    monkeypatch.setattr(mw.MainWindow, "collect_monitor_data", lambda self: {})
    monkeypatch.setattr(mw.MainWindow, "_log_follower_active", lambda self: False)
    calls = []
    monkeypatch.setattr(mw.MainWindow, "_start_log_follower", lambda self: calls.append(1))
    w = mw.MainWindow()
    qtbot.addWidget(w)
    calls.clear()
    w.update_status()
    assert calls == [1]


def test_update_status_does_not_restart_active_follower(qtbot, monkeypatch):
    """A follower already streaming must not be re-spawned on every poll."""
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    monkeypatch.setattr(mw.runtime, "container_state", lambda name, binary: "running")
    monkeypatch.setattr(mw.health, "health_ok", lambda port: True)
    monkeypatch.setattr(mw.MainWindow, "collect_monitor_data", lambda self: {})
    monkeypatch.setattr(mw.MainWindow, "_log_follower_active", lambda self: True)
    calls = []
    monkeypatch.setattr(mw.MainWindow, "_start_log_follower", lambda self: calls.append(1))
    w = mw.MainWindow()
    qtbot.addWidget(w)
    calls.clear()
    w.update_status()
    assert calls == []


def test_on_launch_invokes_terminal(qtbot, monkeypatch):
    captured = {}
    monkeypatch.setattr(mw.terminal, "launch",
                        lambda argv, template=mw.terminal.DEFAULT_TEMPLATE:
                        captured.setdefault("argv", argv))
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w.load_profile(_profile())
    w.on_launch()
    assert captured["argv"][0] == "podman"
    assert "img:tag" in captured["argv"]


def test_on_launch_blocks_on_validation_error(qtbot, monkeypatch):
    called = {"launched": False}
    monkeypatch.setattr(mw.terminal, "launch",
                        lambda *a, **k: called.__setitem__("launched", True))
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    p = _profile(); p.model = ""  # invalid
    w.load_profile(p)
    w.on_launch()
    assert called["launched"] is False


def test_fetch_latest_updates_image(qtbot, monkeypatch):
    monkeypatch.setattr(mw.registry, "fetch_latest",
                        lambda repo, prefix, timeout=10.0: "server-cuda12-b9999")
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w.image_edit.setText("ghcr.io/ggml-org/llama.cpp:server-cuda12-b1")
    w.on_fetch_latest()
    assert w.image_edit.text() == "ghcr.io/ggml-org/llama.cpp:server-cuda12-b9999"
