import llama_launcher.ui.main_window as mw


def test_detect_image_single_fills(qtbot, monkeypatch):
    monkeypatch.setattr(mw.runtime, "list_local_images",
                        lambda b, engine="llama.cpp": ["ghcr.io/ggml-org/llama.cpp:full"])
    w = mw.MainWindow(); qtbot.addWidget(w)
    w.image_edit.setText("")
    w.detect_image()
    assert w.image_edit.text() == "ghcr.io/ggml-org/llama.cpp:full"


def test_detect_image_multiple_prompts(qtbot, monkeypatch):
    imgs = ["ghcr.io/ggml-org/llama.cpp:full", "ghcr.io/ggml-org/llama.cpp:server"]
    monkeypatch.setattr(mw.runtime, "list_local_images", lambda b, engine="llama.cpp": imgs)
    monkeypatch.setattr(mw.QInputDialog, "getItem",
                        staticmethod(lambda *a, **k: (imgs[1], True)))
    w = mw.MainWindow(); qtbot.addWidget(w)
    w.detect_image()
    assert w.image_edit.text() == imgs[1]


def test_detect_image_none_keeps_field(qtbot, monkeypatch):
    monkeypatch.setattr(mw.runtime, "list_local_images", lambda b, engine="llama.cpp": [])
    monkeypatch.setattr(mw.QMessageBox, "information", lambda *a, **k: None)
    w = mw.MainWindow(); qtbot.addWidget(w)
    w.image_edit.setText("keep-this")
    w.detect_image()
    assert w.image_edit.text() == "keep-this"


def test_autofill_image_on_empty_startup(qtbot, monkeypatch):
    monkeypatch.setattr(mw.runtime, "list_local_images",
                        lambda b, engine="llama.cpp": ["ghcr.io/ggml-org/llama.cpp:full"])
    w = mw.MainWindow(); qtbot.addWidget(w)
    assert w.image_edit.text() == "ghcr.io/ggml-org/llama.cpp:full"


def test_no_autofill_when_multiple(qtbot, monkeypatch):
    monkeypatch.setattr(mw.runtime, "list_local_images",
                        lambda b, engine="llama.cpp": ["a/llama.cpp:full", "a/llama.cpp:server"])
    w = mw.MainWindow(); qtbot.addWidget(w)
    assert w.image_edit.text() == ""   # ambiguous -> not auto-filled


def test_detect_uses_selected_engine(qtbot, monkeypatch):
    seen = {}

    def fake_list(binary, engine="llama.cpp"):
        seen["engine"] = engine
        return ["ghcr.io/ikawrakow/ik-llama-cpp:cu12-server"]

    monkeypatch.setattr(mw.runtime, "list_local_images", fake_list)
    w = mw.MainWindow(); qtbot.addWidget(w)
    w.engine_combo.setCurrentIndex(w.engine_combo.findData("ik_llama.cpp"))
    w.detect_image()
    assert seen["engine"] == "ik_llama.cpp"
