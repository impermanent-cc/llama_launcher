import pytest

from llama_launcher.core.spec import Profile
from llama_launcher.store import profiles as store
from llama_launcher.ui.main_window import MainWindow


@pytest.fixture
def win(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _router_index(win):
    for i in range(win.mode_combo.count()):
        if win.mode_combo.itemData(i) == "router":
            return i
    raise AssertionError("no router mode in combo")


def test_switching_to_router_mode_populates_the_key(win):
    # A member profile must exist so current_profile() builds a valid router.
    store.save_profile(Profile(name="gen", image="img", settings={"port": 8080}),
                       store.default_base_dir())
    win.name_edit.setText("myrouter")
    win._add_member_item(win._member_from_name("gen") if hasattr(win, "_member_from_name")
                         else __import__("llama_launcher.core.spec", fromlist=["RouterMember"]).RouterMember(profile="gen"))
    win.mode_combo.setCurrentIndex(_router_index(win))   # fires _on_mode_changed
    assert win.router_panel._api_key, "key should be generated on entering router mode"
    win.router_panel.reveal_check.setChecked(True)
    assert win.router_panel.key_label.text() == win.router_panel._api_key


def test_showing_router_tab_populates_the_key(win):
    store.save_profile(Profile(name="gen", image="img", settings={"port": 8080}),
                       store.default_base_dir())
    win.name_edit.setText("r2")
    from llama_launcher.core.spec import RouterMember
    win._add_member_item(RouterMember(profile="gen"))
    win.mode_combo.setCurrentIndex(_router_index(win))
    win.router_panel._api_key = ""            # simulate a stale/empty panel
    # switch to the Router tab -> _on_tab_changed should refresh the header
    for i in range(win.tabs.count()):
        if win.tabs.widget(i) is win.router_panel:
            win.tabs.setCurrentIndex(i)
            break
    assert win.router_panel._api_key, "key should repopulate when the Router tab is shown"


from llama_launcher.services import api_key


def _make_router(win, name):
    from llama_launcher.core.spec import RouterMember
    store.save_profile(Profile(name="gen", image="img", settings={"port": 8080}),
                       store.default_base_dir())
    win.name_edit.setText(name)
    win._add_member_item(RouterMember(profile="gen"))
    win.mode_combo.setCurrentIndex(_router_index(win))


def test_scope_radio_flows_into_current_profile(win):
    _make_router(win, "r_mode")
    win.router_panel.scope_own.setChecked(True)     # fires key_scope_changed
    assert win.current_profile().runtime.router_key_mode == "own"


def test_saving_global_key_writes_global_file(win):
    _make_router(win, "r_glob")
    win.router_panel.set_scope("global")
    win.router_panel._save_key("sk-shared")         # fires key_saved
    assert api_key.read_global_key(win.router_base_dir()) == "sk-shared"


def test_saving_own_key_writes_per_profile_file(win):
    _make_router(win, "r_own")
    win.router_panel.set_scope("own")
    win.router_panel._save_key("sk-mine")
    assert api_key.read_api_key(win.router_base_dir(), "r_own") == "sk-mine"


def test_saving_key_while_router_running_warns_a_relaunch_is_needed(win, monkeypatch):
    from llama_launcher.ui import main_window as mw

    _make_router(win, "r_running")
    monkeypatch.setattr(mw.runtime, "container_state", lambda name, binary: "running")
    calls = []
    monkeypatch.setattr(mw.QMessageBox, "information",
                        lambda *a, **k: calls.append((a, k)))
    win.router_panel.set_scope("global")
    win.router_panel._save_key("sk-shared")
    assert len(calls) == 1


@pytest.mark.parametrize("state", ["absent", "stopped"])
def test_saving_key_while_router_not_running_does_not_warn(win, monkeypatch, state):
    from llama_launcher.ui import main_window as mw

    _make_router(win, "r_idle")
    monkeypatch.setattr(mw.runtime, "container_state", lambda name, binary: state)
    calls = []
    monkeypatch.setattr(mw.QMessageBox, "information",
                        lambda *a, **k: calls.append((a, k)))
    win.router_panel.set_scope("global")
    win.router_panel._save_key("sk-shared")
    assert calls == []


def test_loading_profile_syncs_scope_radio(win):
    from llama_launcher.core.spec import Runtime, RouterMember
    store.save_profile(Profile(name="gen", image="img", settings={"port": 8080}),
                       store.default_base_dir())
    p = Profile(name="r_load", image="img", mode="router",
                runtime=Runtime(router_key_mode="own"), settings={"port": 8080},
                members=[RouterMember(profile="gen")])
    win.load_profile(p)
    assert win.router_panel._current_scope() == "own"
