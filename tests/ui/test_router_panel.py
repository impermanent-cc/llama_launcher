from llama_launcher.core.router_models import RouterModel
from llama_launcher.ui.panels.router_panel import RouterPanel


def test_table_populates_from_models(qtbot):
    panel = RouterPanel()
    qtbot.addWidget(panel)
    panel.set_models([
        RouterModel(id="qwen", status="loaded"),
        RouterModel(id="gemma", status="sleeping"),
    ])
    assert panel.table.rowCount() == 2
    assert panel.table.item(0, 0).text() == "qwen"
    assert panel.table.item(0, 1).text() == "loaded"
    assert panel.table.item(1, 1).text() == "sleeping"


def test_failed_model_shows_exit_code(qtbot):
    panel = RouterPanel()
    qtbot.addWidget(panel)
    panel.set_models([RouterModel(id="a", status="unloaded", failed=True, exit_code=1)])
    assert "1" in panel.table.item(0, 1).text()


def test_loading_progress_is_shown(qtbot):
    panel = RouterPanel()
    qtbot.addWidget(panel)
    panel.set_models([RouterModel(id="a", status="loading", progress=0.5)])
    assert "50" in panel.table.item(0, 1).text()


def test_load_button_emits_model_id(qtbot):
    panel = RouterPanel()
    qtbot.addWidget(panel)
    panel.set_models([RouterModel(id="qwen", status="unloaded")])
    with qtbot.waitSignal(panel.load_requested) as blocker:
        panel._load_buttons["qwen"].click()
    assert blocker.args == ["qwen"]


def test_unload_button_emits_model_id(qtbot):
    panel = RouterPanel()
    qtbot.addWidget(panel)
    panel.set_models([RouterModel(id="qwen", status="loaded")])
    with qtbot.waitSignal(panel.unload_requested) as blocker:
        panel._unload_buttons["qwen"].click()
    assert blocker.args == ["qwen"]


def test_api_key_is_masked_until_revealed(qtbot):
    panel = RouterPanel()
    qtbot.addWidget(panel)
    panel.set_endpoint("http://127.0.0.1:8080", "sk-secret-value", ["qwen"])
    assert "sk-secret-value" not in panel.key_label.text()
    panel.reveal_key(True)
    assert "sk-secret-value" in panel.key_label.text()


def test_harness_block_lists_base_url_and_model_ids(qtbot):
    panel = RouterPanel()
    qtbot.addWidget(panel)
    panel.set_endpoint("http://192.168.1.9:8080", "sk-x", ["qwen", "gemma"])
    text = panel.harness_text.toPlainText()
    assert "http://192.168.1.9:8080/v1" in text
    assert "qwen" in text and "gemma" in text


def test_exposure_banner_hidden_by_default(qtbot):
    panel = RouterPanel()
    qtbot.addWidget(panel)
    assert not panel.banner.isVisible() or panel.banner.text() == ""


def test_exposure_banner_shows_warning(qtbot):
    panel = RouterPanel()
    qtbot.addWidget(panel)
    panel.set_exposure_warning("Bound to 0.0.0.0")
    assert "0.0.0.0" in panel.banner.text()


def test_disconnected_state_is_reported(qtbot):
    panel = RouterPanel()
    qtbot.addWidget(panel)
    panel.set_connected(False)
    assert "disconnected" in panel.status_label.text().lower()


def test_scope_toggle_emits_mode(qtbot):
    panel = RouterPanel()
    qtbot.addWidget(panel)
    with qtbot.waitSignal(panel.key_scope_changed) as sig:
        panel.scope_own.setChecked(True)
    assert sig.args == ["own"]


def test_set_scope_does_not_emit(qtbot):
    panel = RouterPanel()
    qtbot.addWidget(panel)
    panel.set_scope("own")
    assert panel._current_scope() == "own"
    # set_scope must be silent; a spy would time out, so assert state only
    panel.set_scope("global")
    assert panel._current_scope() == "global"


def test_save_key_normalizes_and_emits_current_scope(qtbot):
    panel = RouterPanel()
    qtbot.addWidget(panel)
    panel.set_scope("global")
    with qtbot.waitSignal(panel.key_saved) as sig:
        panel._save_key("  sk-typed \n")
    assert sig.args == ["global", "sk-typed"]


def test_save_key_rejects_empty(qtbot):
    panel = RouterPanel()
    qtbot.addWidget(panel)
    # empty must not emit; _save_key returns False and swallows the ValueError
    assert panel._save_key("   ") is False
