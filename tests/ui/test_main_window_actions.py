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
