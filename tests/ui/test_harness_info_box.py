from llama_launcher.ui.widgets.harness_info_box import HarnessInfoBox


def test_endpoint_lists_models(qtbot):
    box = HarnessInfoBox()
    qtbot.addWidget(box)
    box.set_endpoint("http://127.0.0.1:8080", ["qwen", "gemma"])
    text = box.harness_text.toPlainText()
    assert "http://127.0.0.1:8080/v1" in text
    assert "qwen" in text and "gemma" in text


def test_endpoint_empty_models(qtbot):
    box = HarnessInfoBox()
    qtbot.addWidget(box)
    box.set_endpoint("http://127.0.0.1:8080", [])
    assert "no members yet" in box.harness_text.toPlainText()
