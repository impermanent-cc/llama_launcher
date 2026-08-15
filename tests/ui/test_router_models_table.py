from llama_launcher.core.router_models import RouterModel
from llama_launcher.ui.widgets.router_models_table import RouterModelsTable


def test_table_populates(qtbot):
    t = RouterModelsTable()
    qtbot.addWidget(t)
    t.set_models([RouterModel(id="qwen", status="loaded"),
                  RouterModel(id="gemma", status="sleeping")])
    assert t.table.rowCount() == 2
    assert t.table.item(0, 0).text() == "qwen"
    assert t.table.item(0, 1).text() == "loaded"


def test_load_button_emits_id(qtbot):
    t = RouterModelsTable()
    qtbot.addWidget(t)
    t.set_models([RouterModel(id="qwen", status="unloaded")])
    with qtbot.waitSignal(t.load_requested) as blocker:
        t._load_buttons["qwen"].click()
    assert blocker.args == ["qwen"]


def test_capped_height_for_scroll(qtbot):
    t = RouterModelsTable()
    qtbot.addWidget(t)
    assert t.table.maximumHeight() <= 160    # ~3-4 rows then scrolls


def test_unload_button_emits_id(qtbot):
    t = RouterModelsTable()
    qtbot.addWidget(t)
    t.set_models([RouterModel(id="qwen", status="loaded")])
    with qtbot.waitSignal(t.unload_requested) as blocker:
        t._unload_buttons["qwen"].click()
    assert blocker.args == ["qwen"]


def test_failed_model_shows_exit_code(qtbot):
    t = RouterModelsTable()
    qtbot.addWidget(t)
    t.set_models([RouterModel(id="a", status="unloaded", failed=True, exit_code=1)])
    assert "1" in t.table.item(0, 1).text()


def test_loading_progress_is_shown(qtbot):
    t = RouterModelsTable()
    qtbot.addWidget(t)
    t.set_models([RouterModel(id="a", status="loading", progress=0.5)])
    assert "50" in t.table.item(0, 1).text()
