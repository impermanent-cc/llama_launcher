import llama_launcher.ui.main_window as mw
from llama_launcher.store import profiles as store


def test_stats_dock_exists_hidden_by_default(qtbot):
    w = mw.MainWindow(); qtbot.addWidget(w)
    assert w.stats_dock is not None
    assert w.stats_dock.isVisibleTo(w) is False          # default closed
    assert w.stats_toggle_btn.isChecked() is False


def test_toggle_button_shows_dock(qtbot):
    w = mw.MainWindow(); qtbot.addWidget(w)
    w.stats_toggle_btn.setChecked(True)
    assert w.stats_dock.isVisibleTo(w) is True
    w.stats_toggle_btn.setChecked(False)
    assert w.stats_dock.isVisibleTo(w) is False


def test_open_state_persists(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    w = mw.MainWindow(); qtbot.addWidget(w)
    w.stats_toggle_btn.setChecked(True)
    w._save_stats_config()
    from llama_launcher.store.profiles import load_config, default_base_dir
    assert load_config(default_base_dir()).get("stats_open") is True
    # a fresh window restores the open state
    w2 = mw.MainWindow(); qtbot.addWidget(w2)
    assert w2.stats_dock.isVisibleTo(w2) is True
