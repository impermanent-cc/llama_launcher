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
    for i in range(win._configure_panel.mode_combo.count()):
        if win._configure_panel.mode_combo.itemData(i) == "router":
            return i
    raise AssertionError("no router mode in combo")


def test_switching_to_router_mode_populates_the_key(win):
    # A member profile must exist so current_profile() builds a valid router.
    store.save_profile(Profile(name="gen", image="img", settings={"port": 8080}),
                       store.default_base_dir())
    win._configure_panel.name_edit.setText("myrouter")
    win._configure_panel._add_member_item(win._configure_panel._member_from_name("gen") if hasattr(win._configure_panel, "_member_from_name")
                         else __import__("llama_launcher.core.spec", fromlist=["RouterMember"]).RouterMember(profile="gen"))
    win._configure_panel.mode_combo.setCurrentIndex(_router_index(win))   # fires _on_mode_changed
    assert win._configure_panel.api_key_box._api_key, "key should be generated on entering router mode"
    win._configure_panel.api_key_box.reveal_check.setChecked(True)
    assert win._configure_panel.api_key_box.key_label.text() == win._configure_panel.api_key_box._api_key


def test_switching_to_configure_tab_populates_the_key(win):
    store.save_profile(Profile(name="gen", image="img", settings={"port": 8080}),
                       store.default_base_dir())
    win._configure_panel.name_edit.setText("r2")
    from llama_launcher.core.spec import RouterMember
    win._configure_panel._add_member_item(RouterMember(profile="gen"))
    win._configure_panel.mode_combo.setCurrentIndex(_router_index(win))
    win._configure_panel.api_key_box._api_key = ""             # simulate a stale/empty box
    # switch away then back to Configure -> _on_tab_changed should refresh the header
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    win.tabs.setCurrentIndex(titles.index("Monitor"))
    win.tabs.setCurrentIndex(titles.index("Configure"))
    assert win._configure_panel.api_key_box._api_key, "key should repopulate when the Configure tab is shown"


from llama_launcher.services import api_key


def _make_router(win, name):
    from llama_launcher.core.spec import RouterMember
    store.save_profile(Profile(name="gen", image="img", settings={"port": 8080}),
                       store.default_base_dir())
    win._configure_panel.name_edit.setText(name)
    win._configure_panel._add_member_item(RouterMember(profile="gen"))
    win._configure_panel.mode_combo.setCurrentIndex(_router_index(win))


def test_router_key_ui_hidden_in_server_mode(win):
    # The reusable API key + harness block + status/exposure banner are
    # ROUTER-only. The default startup profile is single-server, so they must
    # not show there (a single server uses the --api-key field in Settings).
    assert win._configure_panel.mode_combo.currentData() == "server"
    assert win._configure_panel.api_key_box.isHidden()
    assert win._configure_panel.harness_box.isHidden()
    assert win._configure_panel.configure_status.isHidden()
    assert win.monitor_status.isHidden()


def test_router_key_ui_shown_in_router_mode(win):
    _make_router(win, "r_vis")
    assert not win._configure_panel.api_key_box.isHidden()
    assert not win._configure_panel.harness_box.isHidden()
    assert not win._configure_panel.configure_status.isHidden()


def test_scope_radio_flows_into_current_profile(win):
    _make_router(win, "r_mode")
    win._configure_panel.api_key_box.scope_own.setChecked(True)      # fires key_scope_changed
    assert win._configure_panel.current_profile().runtime.router_key_mode == "own"


def test_saving_global_key_writes_global_file(win):
    _make_router(win, "r_glob")
    win._configure_panel.api_key_box.set_scope("global")
    win._configure_panel.api_key_box._save_key("sk-shared")          # fires key_saved
    assert api_key.read_global_key(win.router_base_dir()) == "sk-shared"


def test_saving_own_key_writes_per_profile_file(win):
    _make_router(win, "r_own")
    win._configure_panel.api_key_box.set_scope("own")
    win._configure_panel.api_key_box._save_key("sk-mine")
    assert api_key.read_api_key(win.router_base_dir(), "r_own") == "sk-mine"


def test_saving_key_while_router_running_warns_a_relaunch_is_needed(win, monkeypatch):
    from llama_launcher.ui import main_window as mw

    _make_router(win, "r_running")
    monkeypatch.setattr(mw.runtime, "container_state", lambda name, binary: "running")
    calls = []
    monkeypatch.setattr(mw.QMessageBox, "information",
                        lambda *a, **k: calls.append((a, k)))
    win._configure_panel.api_key_box.set_scope("global")
    win._configure_panel.api_key_box._save_key("sk-shared")
    assert len(calls) == 1


@pytest.mark.parametrize("state", ["absent", "stopped"])
def test_saving_key_while_router_not_running_does_not_warn(win, monkeypatch, state):
    from llama_launcher.ui import main_window as mw

    _make_router(win, "r_idle")
    monkeypatch.setattr(mw.runtime, "container_state", lambda name, binary: state)
    calls = []
    monkeypatch.setattr(mw.QMessageBox, "information",
                        lambda *a, **k: calls.append((a, k)))
    win._configure_panel.api_key_box.set_scope("global")
    win._configure_panel.api_key_box._save_key("sk-shared")
    assert calls == []


def test_loading_profile_syncs_scope_radio(win):
    from llama_launcher.core.spec import Runtime, RouterMember
    store.save_profile(Profile(name="gen", image="img", settings={"port": 8080}),
                       store.default_base_dir())
    p = Profile(name="r_load", image="img", mode="router",
                runtime=Runtime(router_key_mode="own"), settings={"port": 8080},
                members=[RouterMember(profile="gen")])
    win._configure_panel.load_profile(p)
    assert win._configure_panel.api_key_box._current_scope() == "own"
