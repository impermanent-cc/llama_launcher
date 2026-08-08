import pytest

from llama_launcher.core.presets import Preset
from llama_launcher.core.spec import Mount, Profile, RouterMember
from llama_launcher.ui.main_window import MainWindow


@pytest.fixture
def win(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _chip_texts(win):
    texts = []
    for i in range(win._suggestions_layout.count()):
        wdg = win._suggestions_layout.itemAt(i).widget()
        if wdg is not None:
            texts.append(wdg.text())
    return texts


def _pick_family(win, preset):
    # Drive the picker the way the UI does, then invoke its handler directly.
    win.family_combo.addItem(preset.label, preset)
    win.family_combo.setCurrentIndex(win.family_combo.count() - 1)
    win._on_pick_family(win.family_combo.currentIndex())


def test_selecting_a_family_shows_its_chips_without_a_model(win):
    win.load_profile(Profile(name="Solo", image="img", settings={"port": 8080}))
    _pick_family(win, Preset(key="fam", label="Fam", settings={"temp": 0.6}))
    texts = _chip_texts(win)
    assert any("temp = 0.6" in t for t in texts)
    assert any("Apply all Fam defaults" in t for t in texts)


def test_clicking_a_per_option_chip_applies_only_that_option(win):
    # top-k's catalog default is 40, and the settings-form contract only
    # round-trips values that differ from default (see SettingWidget.is_set);
    # 64 (not 40) is used here so "untouched" is actually observable.
    win.load_profile(Profile(name="Solo", image="img",
                             settings={"port": 8080, "top-k": 64}))
    _pick_family(win, Preset(key="fam", label="Fam", settings={"temp": 0.6}))
    win._apply_suggestion(  # what the chip's click calls
        next(s for s in __import__("llama_launcher.core.presets",
                                   fromlist=["preset_suggestions"])
             .preset_suggestions(win._preset_family) if s.text == "temp = 0.6"))
    p = win.current_profile()
    assert p.settings["temp"] == 0.6         # applied
    assert p.settings["top-k"] == 64         # untouched


def test_none_family_clears_preset_chips(win):
    win.load_profile(Profile(name="Solo", image="img", settings={"port": 8080}))
    _pick_family(win, Preset(key="fam", label="Fam", settings={"temp": 0.6}))
    assert any("temp = 0.6" in t for t in _chip_texts(win))
    win._preset_family = None
    win._rebuild_suggestions(win._last_caps)
    assert not any("temp = 0.6" in t for t in _chip_texts(win))


def test_router_mode_hides_server_only_preset_chips(win):
    # temp is a server-only setting; a router profile must not show it.
    win.load_profile(Profile(name="Host", mode="router", image="img",
                             members=[RouterMember(profile="Q")]))
    _pick_family(win, Preset(key="fam", label="Fam", settings={"temp": 0.6}))
    assert not any("temp = 0.6" in t for t in _chip_texts(win))


from llama_launcher.store.presets import list_presets


def test_save_as_preset_captures_set_options(win, tmp_path, monkeypatch):
    from llama_launcher.store.profiles import default_base_dir
    win.load_profile(Profile(name="Solo", image="img",
                             settings={"port": 8080, "temp": 0.6, "top-k": 20}))
    monkeypatch.setattr("llama_launcher.ui.main_window.QInputDialog.getText",
                        lambda *a, **k: ("My Fam", True))
    win.on_save_preset()
    saved = list_presets(default_base_dir())
    assert [p.label for p in saved] == ["My Fam"]
    got = saved[0].settings
    assert got["temp"] == 0.6 and got["top-k"] == 20
    assert "port" not in got                       # port excluded
    assert saved[0].source == "user"
    # ...and it becomes selectable in the picker.
    assert "My Fam" in [win.family_combo.itemText(i)
                        for i in range(win.family_combo.count())]


def test_save_as_preset_noop_when_nothing_set(win, monkeypatch):
    win.load_profile(Profile(name="Solo", image="img", settings={"port": 8080}))
    monkeypatch.setattr("llama_launcher.ui.main_window.QInputDialog.getText",
                        lambda *a, **k: ("Empty", True))
    seen = {}
    monkeypatch.setattr("llama_launcher.ui.main_window.QMessageBox.information",
                        lambda *a, **k: seen.setdefault("info", True))
    win.on_save_preset()
    from llama_launcher.store.profiles import default_base_dir
    from llama_launcher.store.presets import list_presets as lp
    assert lp(default_base_dir()) == []            # nothing saved
    assert seen.get("info")                        # user was told
