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
