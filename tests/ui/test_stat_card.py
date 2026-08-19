from llama_launcher.ui.widgets.stat_card import StatCard


def _row(**over):
    r = {"profile": "gen", "port": 8080, "health": "ready", "tok_s": 64.0,
         "kv_pct": 0.5, "embeddings": False, "reranking": False, "mode": "server",
         "running": True}
    r.update(over)
    return r


def test_gen_card_shows_tok_s_and_kv(qtbot):
    c = StatCard("llama-gen"); qtbot.addWidget(c)
    c.update_row(_row())
    assert "64 tok/s" in c.headline_text()
    assert "KV 50%" in c.kv_text()


def test_embedding_card_shows_ready(qtbot):
    c = StatCard("llama-emb"); qtbot.addWidget(c)
    c.update_row(_row(embeddings=True, tok_s=None, kv_pct=None))
    assert c.headline_text() == "ready"


def test_router_card_shows_router(qtbot):
    c = StatCard("llama-r"); qtbot.addWidget(c)
    c.update_row(_row(mode="router", tok_s=None, kv_pct=None))
    assert c.headline_text() == "router"


def test_stop_button_emits_name(qtbot):
    c = StatCard("llama-gen"); qtbot.addWidget(c)
    got = []; c.stop_requested.connect(got.append)
    c.stop_button().click()
    assert got == ["llama-gen"]


def test_set_selected_toggles(qtbot):
    c = StatCard("x"); qtbot.addWidget(c)
    c.set_selected(True); assert c.is_selected()
    c.set_selected(False); assert not c.is_selected()


def test_running_card_stop_button_emits_stop_requested(qtbot):
    c = StatCard("llama-gen"); qtbot.addWidget(c)
    c.update_row(_row(running=True))
    got = []; c.stop_requested.connect(got.append)
    c.stop_button().click()
    assert got == ["llama-gen"]


def test_stopped_card_action_button_offers_remove(qtbot):
    c = StatCard("llama-dead"); qtbot.addWidget(c)
    c.update_row(_row(running=False, health="down"))
    assert c.stop_button().text() == "✕"
    got = []; c.remove_requested.connect(got.append)
    stop_got = []; c.stop_requested.connect(stop_got.append)
    c.stop_button().click()
    assert got == ["llama-dead"]
    assert stop_got == []
