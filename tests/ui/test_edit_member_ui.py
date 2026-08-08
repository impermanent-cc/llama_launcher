import pytest

from llama_launcher.core.spec import Profile, RouterMember
from llama_launcher.store import profiles as store
from llama_launcher.ui.main_window import MainWindow


@pytest.fixture
def win(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _saved_router_with_member(win):
    base = store.default_base_dir()
    store.save_profile(Profile(name="gen", image="img",
                               settings={"port": 8080, "n-gpu-layers": 99}), base)
    router = Profile(name="myrouter", mode="router", image="img",
                     settings={"port": 9000}, members=[RouterMember(profile="gen")])
    store.save_profile(router, base)          # saved => clean
    win.load_profile(router)
    win.members_list.setCurrentCell(0, 0)


def test_edit_member_on_clean_router_loads_member_no_prompt(win, monkeypatch):
    _saved_router_with_member(win)
    called = {"q": 0}
    monkeypatch.setattr("llama_launcher.ui.main_window.QMessageBox.question",
                        lambda *a, **k: called.__setitem__("q", called["q"] + 1))
    win._on_edit_member()
    assert called["q"] == 0                     # clean => no prompt
    assert win.current_profile().name == "gen"  # form now shows the member
    assert win.current_profile().mode != "router"


def test_edit_member_with_unsaved_changes_prompts_and_cancel_aborts(win, monkeypatch):
    _saved_router_with_member(win)
    win.name_edit.setText("myrouter-edited")    # unsaved change to the router
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr("llama_launcher.ui.main_window.QMessageBox.question",
                        lambda *a, **k: QMessageBox.Cancel)
    win._on_edit_member()
    assert win.current_profile().name == "myrouter-edited"   # stayed on the router


def test_edit_member_with_unsaved_changes_discard_loads_member(win, monkeypatch):
    _saved_router_with_member(win)
    win.name_edit.setText("myrouter-edited")
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr("llama_launcher.ui.main_window.QMessageBox.question",
                        lambda *a, **k: QMessageBox.Discard)
    win._on_edit_member()
    assert win.current_profile().name == "gen"


def test_edit_member_missing_profile_warns(win, monkeypatch):
    base = store.default_base_dir()
    router = Profile(name="r", mode="router", image="img",
                     settings={"port": 9000}, members=[RouterMember(profile="ghost")])
    store.save_profile(router, base)
    win.load_profile(router)
    win.members_list.setCurrentCell(0, 0)
    seen = {}
    monkeypatch.setattr("llama_launcher.ui.main_window.QMessageBox.warning",
                        lambda *a, **k: seen.setdefault("warned", True))
    win._on_edit_member()
    assert seen.get("warned")
    assert win.current_profile().name == "r"    # unchanged


def test_members_guidance_label_present(win):
    from llama_launcher.core.spec import Profile as P
    win.load_profile(P(name="r", mode="router", image="img", settings={"port": 9000}))
    assert win.members_guidance.text().strip()
    assert win.members_guidance.isVisibleTo(win.members_guidance.parentWidget())
