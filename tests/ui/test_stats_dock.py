import llama_launcher.ui.main_window as mw


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


def test_stats_worker_emits_then_stops(qtbot):
    from llama_launcher.ui.main_window import StatsWorker
    from llama_launcher.services.stats import StatsSnapshot
    snap = StatsSnapshot(gpus=[], cpu=None, mem=None, container=None, gpu_available=False)
    w = StatsWorker(lambda: snap, interval_ms=50)
    with qtbot.waitSignal(w.sampled, timeout=2000) as blocker:
        w.start()
    assert blocker.args[0] is snap
    w.stop()
    assert w.wait(2000) is True


def test_opening_dock_starts_worker_and_teardown_drains_it(qtbot):
    w = mw.MainWindow(); qtbot.addWidget(w)
    w.stats_toggle_btn.setChecked(True)          # visible -> worker starts
    assert w._stats_worker is not None and w._stats_worker.isRunning()
    w._stop_timers()                             # teardown must join it
    assert not w._stats_worker.isRunning()
