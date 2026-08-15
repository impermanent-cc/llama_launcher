from llama_launcher.ui.widgets.status_banner import StatusBanner


def test_exposure_warning_toggles_visibility(qtbot):
    b = StatusBanner()
    qtbot.addWidget(b); b.show()
    b.set_exposure_warning("Bound to 0.0.0.0")
    assert b.banner.isVisible() and "0.0.0.0" in b.banner.text()
    b.set_exposure_warning("")
    assert not b.banner.isVisible()


def test_connected_and_error_text(qtbot):
    b = StatusBanner()
    qtbot.addWidget(b)
    b.set_connected(True)
    assert "connected" in b.status_label.text()
    b.set_error("load failed: qwen")
    assert "load failed: qwen" in b.status_label.text()
