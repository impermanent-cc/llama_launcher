import llama_launcher.ui.main_window as mw
from llama_launcher.ui.controllers import launch_controller


def test_detect_image_single_fills(qtbot, monkeypatch):
    monkeypatch.setattr(mw.runtime, "list_local_images",
                        lambda b, engine="llama.cpp": ["ghcr.io/ggml-org/llama.cpp:full"])
    w = mw.MainWindow(); qtbot.addWidget(w)
    w._configure_panel.image_edit.setText("")
    w._launch.detect_image()
    assert w._configure_panel.image_edit.text() == "ghcr.io/ggml-org/llama.cpp:full"


def test_detect_image_multiple_prompts(qtbot, monkeypatch):
    imgs = ["ghcr.io/ggml-org/llama.cpp:full", "ghcr.io/ggml-org/llama.cpp:server"]
    monkeypatch.setattr(mw.runtime, "list_local_images", lambda b, engine="llama.cpp": imgs)
    monkeypatch.setattr(launch_controller.QInputDialog, "getItem",
                        staticmethod(lambda *a, **k: (imgs[1], True)))
    w = mw.MainWindow(); qtbot.addWidget(w)
    w._launch.detect_image()
    assert w._configure_panel.image_edit.text() == imgs[1]


def test_detect_image_none_keeps_field(qtbot, monkeypatch):
    monkeypatch.setattr(mw.runtime, "list_local_images", lambda b, engine="llama.cpp": [])
    monkeypatch.setattr(launch_controller.QMessageBox, "information", lambda *a, **k: None)
    w = mw.MainWindow(); qtbot.addWidget(w)
    w._configure_panel.image_edit.setText("keep-this")
    w._launch.detect_image()
    assert w._configure_panel.image_edit.text() == "keep-this"


def test_autofill_image_on_empty_startup(qtbot, monkeypatch):
    monkeypatch.setattr(mw.runtime, "list_local_images",
                        lambda b, engine="llama.cpp": ["ghcr.io/ggml-org/llama.cpp:full"])
    w = mw.MainWindow(); qtbot.addWidget(w)
    assert w._configure_panel.image_edit.text() == "ghcr.io/ggml-org/llama.cpp:full"


def test_no_autofill_when_multiple(qtbot, monkeypatch):
    monkeypatch.setattr(mw.runtime, "list_local_images",
                        lambda b, engine="llama.cpp": ["a/llama.cpp:full", "a/llama.cpp:server"])
    w = mw.MainWindow(); qtbot.addWidget(w)
    assert w._configure_panel.image_edit.text() == ""   # ambiguous -> not auto-filled


def test_detect_uses_selected_engine(qtbot, monkeypatch):
    seen = {}

    def fake_list(binary, engine="llama.cpp"):
        seen["engine"] = engine
        return ["ghcr.io/ikawrakow/ik-llama-cpp:cu12-server"]

    monkeypatch.setattr(mw.runtime, "list_local_images", fake_list)
    w = mw.MainWindow(); qtbot.addWidget(w)
    w._configure_panel.engine_combo.setCurrentIndex(w._configure_panel.engine_combo.findData("ik_llama.cpp"))
    w._launch.detect_image()
    assert seen["engine"] == "ik_llama.cpp"
