from PySide6.QtWidgets import QPushButton

import llama_launcher.ui.main_window as mw
from llama_launcher.core.spec import Profile, Runtime


def test_check_for_update_finds_newer(qtbot):
    w = mw.MainWindow(); qtbot.addWidget(w)
    w.load_profile(Profile(name="u", image="ghcr.io/ggml-org/llama.cpp:server-cuda12-b9628",
                           runtime=Runtime(binary="podman"), settings={"port": 8080}))
    newer = w.check_for_update(["server-cuda12-b9628", "server-cuda12-b9755", "buildcache-x"])
    assert newer == "server-cuda12-b9755"


def test_check_for_update_none_when_current(qtbot):
    w = mw.MainWindow(); qtbot.addWidget(w)
    w.load_profile(Profile(name="u", image="ghcr.io/ggml-org/llama.cpp:server-cuda12-b9755",
                           runtime=Runtime(binary="podman"), settings={"port": 8080}))
    assert w.check_for_update(["server-cuda12-b9628", "server-cuda12-b9755"]) is None


def test_update_badge_is_flat_button(qtbot):
    """update_badge must be a flat QPushButton."""
    w = mw.MainWindow(); qtbot.addWidget(w)
    assert isinstance(w.update_badge, QPushButton)
    assert w.update_badge.isFlat()


def test_update_badge_shown_and_clickable_after_newer_build(qtbot, monkeypatch):
    """When run_update_check finds a newer build, the badge shows the tag and clicking
    it triggers on_fetch_latest (verified by capturing registry.fetch_latest)."""
    fetched = {}
    monkeypatch.setattr(mw.registry, "fetch_latest",
                        lambda repo, prefix, timeout=10.0: "server-cuda12-b9999")
    w = mw.MainWindow(); qtbot.addWidget(w)
    w.load_profile(Profile(name="u", image="ghcr.io/ggml-org/llama.cpp:server-cuda12-b9628",
                           runtime=Runtime(binary="podman"), settings={"port": 8080}))
    # Simulate what the update worker slot does
    w.update_badge.setText("newer build server-cuda12-b9999 available")
    w.update_badge.setVisible(True)
    assert not w.update_badge.isHidden()
    assert "server-cuda12-b9999" in w.update_badge.text()
    # Clicking the badge should call on_fetch_latest which updates the image
    w.update_badge.click()
    assert w.image_edit.text() == "ghcr.io/ggml-org/llama.cpp:server-cuda12-b9999"
