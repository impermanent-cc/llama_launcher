from llama_launcher.ui.widgets.api_key_box import ApiKeyBox


def test_reveal_shows_key(qtbot):
    box = ApiKeyBox()
    qtbot.addWidget(box)
    box.set_key("sk-secret")
    assert "secret" not in box.key_label.text()  # masked by default
    box.reveal_check.setChecked(True)
    assert box.key_label.text() == "sk-secret"


def test_scope_toggle_emits_once(qtbot):
    box = ApiKeyBox()
    qtbot.addWidget(box)
    with qtbot.waitSignal(box.key_scope_changed) as blocker:
        box.scope_own.setChecked(True)
    assert blocker.args == ["own"]


def test_set_scope_does_not_emit(qtbot):
    box = ApiKeyBox()
    qtbot.addWidget(box)
    fired = []
    box.key_scope_changed.connect(fired.append)
    box.set_scope("own")
    assert fired == []  # programmatic set is silent
    assert box._current_scope() == "own"


def test_save_key_rejects_empty(qtbot):
    box = ApiKeyBox()
    qtbot.addWidget(box)
    seen = []
    box.key_saved.connect(lambda *args: seen.append(args))
    assert box._save_key("   ") is False
    assert seen == []
