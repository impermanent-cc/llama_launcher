from pathlib import Path

import pytest

from llama_launcher.core.spec import Mount, Profile, RouterMember, Runtime
from llama_launcher.store import profiles as store
from llama_launcher.ui.main_window import MainWindow


@pytest.fixture
def win(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def win_base(win):
    from llama_launcher.ui.main_window import base_dir
    return base_dir()


def _member_profile(base, name="Qwen"):
    p = Profile(name=name, image="img", model="/models/qwen.gguf",
                mounts=[Mount(host="/mnt/models", container="/models")],
                settings={"ctx-size": 8192})
    store.save_profile(p, base)
    return p


def test_router_controls_relocated_to_configure_and_monitor(win):
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert "Router" not in titles
    assert win._configure_panel.api_key_box is not None
    assert win._configure_panel.harness_box is not None
    assert win.router_models_table is not None


def test_mode_round_trips_through_the_form(win):
    p = Profile(name="Host", mode="router", image="img",
                members=[RouterMember(profile="Qwen", model_id="qwen")])
    win._configure_panel.load_profile(p)
    back = win._configure_panel.current_profile()
    assert back.mode == "router"
    assert back.members == [RouterMember(profile="Qwen", model_id="qwen")]


def test_bind_host_round_trips(win):
    p = Profile(name="Host", mode="router", image="img",
                runtime=Runtime(bind_host="0.0.0.0"))
    win._configure_panel.load_profile(p)
    assert win._configure_panel.current_profile().runtime.bind_host == "0.0.0.0"


def test_prepare_router_files_writes_preset_and_key(win, tmp_path):
    base = store.default_base_dir()
    _member_profile(base)
    win._configure_panel.load_profile(Profile(name="Host", mode="router", image="img",
                             mounts=[Mount(host="/mnt/models", container="/models")],
                             members=[RouterMember(profile="Qwen")]))
    router_dir, warnings = win.prepare_router_files()
    assert (Path(router_dir) / "models.ini").exists()
    assert (Path(router_dir) / "api-key").exists()
    assert "[qwen]" in (Path(router_dir) / "models.ini").read_text()
    assert warnings == []


def test_router_launch_uses_detached_command(win, monkeypatch, tmp_path):
    base = store.default_base_dir()
    _member_profile(base)
    win._configure_panel.load_profile(Profile(name="Host", mode="router", image="img",
                             mounts=[Mount(host="/mnt/models", container="/models")],
                             members=[RouterMember(profile="Qwen")],
                             settings={"port": 8080}))
    spawned = []
    monkeypatch.setattr(win._launch, "_validate_or_warn", lambda: True)
    monkeypatch.setattr(win._launch, "vram_check", lambda: None)

    pending = []

    def fake_spawn(argv, on_done=None, on_error=None):
        spawned.append(argv)
        # Do NOT call on_done inline: the real QProcess fires it later, so
        # calling it here would make a chained and an unchained implementation
        # produce identical output, which is exactly what this test must tell
        # apart.
        if on_done is not None:
            pending.append(on_done)

    monkeypatch.setattr(win._launch, "_spawn_async", fake_spawn)
    win._launch.on_launch()

    # Only the removal has been spawned so far; the run must be waiting on it.
    assert [a[:3] for a in spawned] == [["podman", "rm", "-f"]]
    assert pending, "the rm must carry a completion callback to chain the run"
    pending.pop(0)()            # rm finishes -> run starts

    run_argv = next(a for a in spawned if "run" in a)
    assert "-d" in run_argv
    assert "--models-preset" in run_argv
    assert ":/router:ro" in " ".join(run_argv)


def test_server_launch_still_uses_the_terminal(win, monkeypatch):
    win._configure_panel.load_profile(Profile(name="Solo", image="img", model="/models/a.gguf",
                             mounts=[Mount(host="/h", container="/models")],
                             settings={"port": 8080}))
    called = {}
    monkeypatch.setattr(win._launch, "_validate_or_warn", lambda: True)
    monkeypatch.setattr(win._launch, "vram_check", lambda: None)
    monkeypatch.setattr("llama_launcher.ui.main_window.terminal.launch",
                        lambda argv: called.setdefault("argv", argv))
    win._launch.on_launch()
    assert "-d" not in called["argv"]


def test_detached_server_launch_uses_spawn_chain_not_terminal(win, monkeypatch):
    win._configure_panel.load_profile(Profile(name="Solo", image="img", model="/models/a.gguf",
                             runtime=Runtime(detached=True),
                             mounts=[Mount(host="/h", container="/models")],
                             settings={"port": 8080}))
    monkeypatch.setattr(win._launch, "_validate_or_warn", lambda: True)
    monkeypatch.setattr(win._launch, "vram_check", lambda: None)
    term_called = {}
    monkeypatch.setattr("llama_launcher.ui.main_window.terminal.launch",
                        lambda argv: term_called.setdefault("argv", argv))
    # A detached server's on_error pops a real QMessageBox (fix #2); stub it
    # so invoking the callback below can't block the offscreen test run.
    monkeypatch.setattr("llama_launcher.ui.controllers.launch_controller.QMessageBox.critical",
                        lambda *a, **k: None)

    spawned, pending, errbacks = [], [], []

    def fake_spawn(argv, on_done=None, on_error=None):
        spawned.append(argv)
        if on_done is not None:
            pending.append(on_done)
        if on_error is not None:
            errbacks.append(on_error)

    monkeypatch.setattr(win._launch, "_spawn_async", fake_spawn)
    win._launch.on_launch()

    # No terminal window in detached mode.
    assert term_called == {}
    # Only the removal has spawned so far; the run waits on it.
    assert [a[:3] for a in spawned] == [["podman", "rm", "-f"]]
    assert pending, "the rm must carry a completion callback to chain the run"
    pending.pop(0)()                      # rm finishes -> run starts

    run_argv = next(a for a in spawned if "run" in a)
    assert "-d" in run_argv
    assert "--rm" not in run_argv
    assert ":/router:ro" not in " ".join(run_argv)   # a server, not a router
    # A detached launch surfaces failures the terminal used to show. The
    # run leg's on_error wraps _report_launch_error(show_dialog=True) --
    # decided here, at launch time -- rather than passing the bare method,
    # so it's checked by behavior (does it report a failure?) rather than
    # identity; test_detached_server_launch_error_pops_a_dialog covers the
    # dialog specifically.
    assert errbacks, "the run must register an on_error callback"
    errbacks[0]("boom")
    assert "failed" in win.status_label.text().lower()


def test_attached_server_launch_still_uses_terminal(win, monkeypatch):
    win._configure_panel.load_profile(Profile(name="Solo", image="img", model="/models/a.gguf",
                             runtime=Runtime(detached=False),
                             mounts=[Mount(host="/h", container="/models")],
                             settings={"port": 8080}))
    monkeypatch.setattr(win._launch, "_validate_or_warn", lambda: True)
    monkeypatch.setattr(win._launch, "vram_check", lambda: None)
    spawned = []
    monkeypatch.setattr(win._launch, "_spawn_async",
                        lambda argv, on_done=None, on_error=None: spawned.append(argv))
    called = {}
    monkeypatch.setattr("llama_launcher.ui.main_window.terminal.launch",
                        lambda argv: called.setdefault("argv", argv))
    win._launch.on_launch()
    assert "-d" not in called["argv"]     # terminal used, attached --rm command
    assert spawned == []                  # no detached chain


def test_preview_reflects_detached_flag(win):
    win._configure_panel.load_profile(Profile(name="Solo", image="img", model="/models/a.gguf",
                             runtime=Runtime(detached=True),
                             mounts=[Mount(host="/h", container="/models")],
                             settings={"port": 8080}))
    argv = win._configure_panel.build_current_command()
    assert "-d" in argv and "--rm" not in argv
    win._configure_panel.detached_check.setChecked(False)
    argv = win._configure_panel.build_current_command()
    assert "--rm" in argv and "-d" not in argv


def test_toggling_detached_checkbox_refreshes_the_preview_widget(win):
    # Unlike test_preview_reflects_detached_flag above (which calls
    # build_current_command() directly, so it can't catch a missing signal
    # connection), this goes through the actual widget/signal path: it
    # toggles the checkbox and reads the preview QPlainTextEdit's own text,
    # the way a user would see it.
    win._configure_panel.load_profile(Profile(name="Solo", image="img", model="/models/a.gguf",
                             runtime=Runtime(detached=False),
                             mounts=[Mount(host="/h", container="/models")],
                             settings={"port": 8080}))
    assert "--rm" in win._configure_panel.preview.toPlainText()
    assert "-d" not in win._configure_panel.preview.toPlainText().split()

    win._configure_panel.detached_check.setChecked(True)

    assert "-d" in win._configure_panel.preview.toPlainText().split()
    assert "--rm" not in win._configure_panel.preview.toPlainText()


def test_detached_server_launch_error_pops_a_dialog(win, monkeypatch):
    win._configure_panel.load_profile(Profile(name="Solo", image="img", model="/models/a.gguf",
                             runtime=Runtime(detached=True),
                             mounts=[Mount(host="/h", container="/models")],
                             settings={"port": 8080}))
    monkeypatch.setattr(win._launch, "_validate_or_warn", lambda: True)
    monkeypatch.setattr(win._launch, "vram_check", lambda: None)
    errbacks = []

    def fake_spawn(argv, on_done=None, on_error=None):
        # Chain immediately: this test only cares which on_error callback
        # the run leg registers, not the rm-then-run ordering (already
        # covered by test_router_launch_uses_detached_command et al).
        if on_error is not None:
            errbacks.append(on_error)
        if on_done is not None:
            on_done()

    monkeypatch.setattr(win._launch, "_spawn_async", fake_spawn)
    dialogs = []
    monkeypatch.setattr("llama_launcher.ui.controllers.launch_controller.QMessageBox.critical",
                        lambda *a, **k: dialogs.append(a))
    win._launch.on_launch()

    assert errbacks, "the run must register an on_error callback"
    errbacks[0]("boom: bad image reference")

    assert dialogs, "a detached server launch failure must pop a QMessageBox"
    assert "bad image reference" in dialogs[0][-1]


def test_router_launch_error_does_not_pop_a_dialog(win, monkeypatch):
    base = store.default_base_dir()
    _member_profile(base)
    win._configure_panel.load_profile(Profile(name="Host", mode="router", image="img",
                             mounts=[Mount(host="/mnt/models", container="/models")],
                             members=[RouterMember(profile="Qwen")],
                             settings={"port": 8080}))
    monkeypatch.setattr(win._launch, "_validate_or_warn", lambda: True)
    monkeypatch.setattr(win._launch, "vram_check", lambda: None)
    errbacks = []

    def fake_spawn(argv, on_done=None, on_error=None):
        # Chain immediately: this test only cares which on_error callback
        # the run leg registers, not the rm-then-run ordering (already
        # covered by test_router_launch_uses_detached_command et al).
        if on_error is not None:
            errbacks.append(on_error)
        if on_done is not None:
            on_done()

    monkeypatch.setattr(win._launch, "_spawn_async", fake_spawn)
    dialogs = []
    monkeypatch.setattr("llama_launcher.ui.controllers.launch_controller.QMessageBox.critical",
                        lambda *a, **k: dialogs.append(a))
    win._launch.on_launch()

    assert errbacks, "the run must register an on_error callback"
    errbacks[0]("boom: bad image reference")

    assert dialogs == [], "router failures must stay on the router panel, no dialog"


def test_detached_server_error_dialog_survives_a_later_mode_switch(win, monkeypatch):
    # _report_launch_error fires from an async QProcess callback, possibly
    # seconds after on_launch(). If show_dialog were re-derived from
    # current_profile().mode at that point (instead of decided once, at the
    # launch call site), switching the mode combo to router while the
    # detached-server launch was still in flight would wrongly suppress its
    # dialog. Prove that doesn't happen: flip the mode AFTER on_launch() but
    # BEFORE invoking the captured on_error.
    win._configure_panel.load_profile(Profile(name="Solo", image="img", model="/models/a.gguf",
                             runtime=Runtime(detached=True),
                             mounts=[Mount(host="/h", container="/models")],
                             settings={"port": 8080}))
    monkeypatch.setattr(win._launch, "_validate_or_warn", lambda: True)
    monkeypatch.setattr(win._launch, "vram_check", lambda: None)
    errbacks = []

    def fake_spawn(argv, on_done=None, on_error=None):
        if on_error is not None:
            errbacks.append(on_error)
        if on_done is not None:
            on_done()

    monkeypatch.setattr(win._launch, "_spawn_async", fake_spawn)
    dialogs = []
    monkeypatch.setattr("llama_launcher.ui.controllers.launch_controller.QMessageBox.critical",
                        lambda *a, **k: dialogs.append(a))
    win._launch.on_launch()
    assert errbacks, "the run must register an on_error callback"

    # The launch is still in flight; the user now switches to a router
    # profile before the (slow) run's error actually arrives.
    win._configure_panel.load_profile(Profile(name="Host", mode="router", image="img",
                             members=[RouterMember(profile="Qwen")]))
    assert win._configure_panel.current_profile().mode == "router"

    errbacks[0]("boom: bad image reference")

    assert dialogs, ("the dialog must still pop -- it was decided at launch "
                     "time for the server launch, not re-derived from the "
                     "now-router UI state")


def test_router_error_dialog_stays_off_after_a_later_mode_switch(win, monkeypatch):
    # Mirror of the above: a router launch's on_error firing after the user
    # has since switched to a server profile must still NOT pop a dialog.
    base = store.default_base_dir()
    _member_profile(base)
    win._configure_panel.load_profile(Profile(name="Host", mode="router", image="img",
                             mounts=[Mount(host="/mnt/models", container="/models")],
                             members=[RouterMember(profile="Qwen")],
                             settings={"port": 8080}))
    monkeypatch.setattr(win._launch, "_validate_or_warn", lambda: True)
    monkeypatch.setattr(win._launch, "vram_check", lambda: None)
    errbacks = []

    def fake_spawn(argv, on_done=None, on_error=None):
        if on_error is not None:
            errbacks.append(on_error)
        if on_done is not None:
            on_done()

    monkeypatch.setattr(win._launch, "_spawn_async", fake_spawn)
    dialogs = []
    monkeypatch.setattr("llama_launcher.ui.controllers.launch_controller.QMessageBox.critical",
                        lambda *a, **k: dialogs.append(a))
    win._launch.on_launch()
    assert errbacks, "the run must register an on_error callback"

    win._configure_panel.load_profile(Profile(name="Solo", image="img", model="/models/a.gguf",
                             mounts=[Mount(host="/h", container="/models")],
                             settings={"port": 8080}))
    assert win._configure_panel.current_profile().mode == "server"

    errbacks[0]("boom: bad image reference")

    assert dialogs == [], ("the router launch's on_error must stay dialog-free "
                           "even though the UI now shows a server profile")


def test_adopt_running_containers_reads_the_label_list(win, monkeypatch):
    monkeypatch.setattr(
        "llama_launcher.ui.main_window.runtime.list_launcher_containers",
        lambda binary: [{"name": "llama-host", "running": True,
                         "profile": "Host", "mode": "router"}])
    rows = win._launch.adopt_running_containers()
    assert rows[0]["profile"] == "Host"


def test_refresh_router_models_populates_the_panel(win, monkeypatch):
    from llama_launcher.core.router_models import RouterModel
    win._configure_panel.load_profile(Profile(name="Host", mode="router", image="img",
                             settings={"port": 8080}))
    monkeypatch.setattr(
        "llama_launcher.ui.main_window.router_api.list_models",
        lambda host, port, key, **kw: [RouterModel(id="qwen", status="sleeping")])
    win._monitor.refresh_router_models()
    assert win.router_models_table.table.rowCount() == 1
    assert win.router_models_table.table.item(0, 0).text() == "qwen"


def test_sleeping_models_are_not_polled_for_metrics(win, monkeypatch):
    """A sleeping model has nothing to measure, and skipping it is the point of
    an idle-unloading host (plan self-review gap 2)."""
    from llama_launcher.core.router_models import RouterModel
    win._configure_panel.load_profile(Profile(name="Host", mode="router", image="img",
                             settings={"port": 8080, "metrics": True}))
    monkeypatch.setattr(
        "llama_launcher.ui.main_window.router_api.list_models",
        lambda host, port, key, **kw: [RouterModel(id="qwen", status="sleeping")])
    win._monitor.refresh_router_models()

    calls = []
    monkeypatch.setattr("llama_launcher.ui.main_window.metrics.fetch_metrics",
                        lambda *a, **kw: calls.append(kw) or {})
    monkeypatch.setattr("llama_launcher.ui.main_window.metrics.fetch_slots",
                        lambda *a, **kw: calls.append(kw) or [])
    win._monitor.collect_monitor_data()
    assert calls == []


def test_loaded_model_is_polled_with_its_id(win, monkeypatch):
    from llama_launcher.core.router_models import RouterModel
    win._configure_panel.load_profile(Profile(name="Host", mode="router", image="img",
                             settings={"port": 8080, "metrics": True}))
    monkeypatch.setattr(
        "llama_launcher.ui.main_window.router_api.list_models",
        lambda host, port, key, **kw: [RouterModel(id="qwen", status="loaded")])
    win._monitor.refresh_router_models()

    seen = {}
    monkeypatch.setattr("llama_launcher.ui.main_window.metrics.fetch_metrics",
                        lambda port, **kw: seen.update(kw) or {})
    monkeypatch.setattr("llama_launcher.ui.main_window.metrics.fetch_slots",
                        lambda port, **kw: [])
    win._monitor.collect_monitor_data()
    assert seen["model"] == "qwen"


def test_router_form_hides_model_level_settings(win):
    win._configure_panel.load_profile(Profile(name="Host", mode="router", image="img"))
    # The precedence trap: router CLI args outrank every member's preset value.
    assert not win._configure_panel._widgets["ctx-size"].isVisibleTo(win._configure_panel.configure_tab)
    assert win._configure_panel._widgets["models-max"].isVisibleTo(win._configure_panel.configure_tab)


def test_server_form_hides_router_only_settings(win):
    win._configure_panel.load_profile(Profile(name="Solo", image="img", model="/m.gguf"))
    assert not win._configure_panel._widgets["models-max"].isVisibleTo(win._configure_panel.configure_tab)
    assert win._configure_panel._widgets["ctx-size"].isVisibleTo(win._configure_panel.configure_tab)


def test_server_profile_cannot_emit_router_only_flags(win):
    from llama_launcher.core.command_builder import build_command
    win._configure_panel.load_profile(Profile(name="Solo", image="img", model="/m.gguf",
                             mounts=[Mount(host="/h", container="/models")],
                             settings={"port": 8080, "models-max": 2}))
    # models-max is router-only; a single-model llama-server rejects it.
    assert "--models-max" not in build_command(win._configure_panel.current_profile())


def _router_win(win, **kw):
    base = dict(name="Host", mode="router", image="img",
                mounts=[Mount(host="/mnt/models", container="/models")],
                members=[RouterMember(profile="Qwen")], settings={"port": 8080})
    base.update(kw)
    win._configure_panel.load_profile(Profile(**base))
    return win


def test_preview_includes_the_router_mount(win):
    _member_profile(store.default_base_dir())
    _router_win(win)
    assert ":/router:ro" in win._configure_panel.preview_text()


def test_exported_script_includes_the_router_mount(win, tmp_path):
    _member_profile(store.default_base_dir())
    _router_win(win)
    out = tmp_path / "run.sh"
    win._report.export_sh(str(out))
    text = out.read_text()
    assert ":/router:ro" in text
    # The preset and key paths are only reachable because of that mount.
    assert "--models-preset" in text


def test_router_header_populates_on_profile_load_not_only_on_launch(win):
    _member_profile(store.default_base_dir())
    _router_win(win)
    # Reattach-after-restart is the common path: select the profile, see the key.
    assert "qwen" in win._configure_panel.harness_box.harness_text.toPlainText()
    win._configure_panel.api_key_box.reveal_key(True)
    assert win._configure_panel.api_key_box.key_label.text().startswith("sk-")


def test_exposure_banner_shows_on_load_for_a_non_loopback_router(win):
    _member_profile(store.default_base_dir())
    _router_win(win, runtime=Runtime(bind_host="0.0.0.0"))
    assert "0.0.0.0" in win._configure_panel.configure_status.banner.text()
    assert "0.0.0.0" in win.monitor_status.banner.text()


def test_exposure_banner_clears_on_switch_to_a_loopback_server(win):
    # Regression: refresh_router_panel_header used to early-return for a
    # non-router profile WITHOUT clearing the relocated router state, so an
    # exposed router's banner (plus its API key and harness endpoint) stayed
    # on screen after switching to an unrelated, perfectly safe loopback
    # server profile. That's a false "you are exposed" warning on the exact
    # surface the spec calls security-critical.
    _member_profile(store.default_base_dir())
    _router_win(win, runtime=Runtime(bind_host="0.0.0.0"))
    # Each tab's own visible flag only reflects reality while it's the
    # current tab (QTabWidget hides the others), so check each banner's
    # setVisible() state with its tab selected.
    win.tabs.setCurrentIndex(0)     # Configure
    assert win._configure_panel.configure_status.banner.isVisibleTo(win)
    win.tabs.setCurrentIndex(1)     # Monitor
    assert win.monitor_status.banner.isVisibleTo(win)
    assert "0.0.0.0" in win._configure_panel.configure_status.banner.text()
    assert "0.0.0.0" in win.monitor_status.banner.text()
    win._configure_panel.api_key_box.reveal_key(True)
    assert win._configure_panel.api_key_box.key_label.text().startswith("sk-")
    assert "qwen" in win._configure_panel.harness_box.harness_text.toPlainText()

    win._configure_panel.load_profile(Profile(name="Solo", image="img", model="/models/a.gguf",
                             mounts=[Mount(host="/h", container="/models")],
                             settings={"port": 8080}))

    assert win._configure_panel.current_profile().mode == "server"
    assert win._configure_panel.configure_status.banner.text() == ""
    assert win.monitor_status.banner.text() == ""
    win.tabs.setCurrentIndex(0)     # Configure
    assert not win._configure_panel.configure_status.banner.isVisibleTo(win)
    win.tabs.setCurrentIndex(1)     # Monitor
    assert not win.monitor_status.banner.isVisibleTo(win)
    # With no key set, "Reveal" shows the mask (there's nothing to reveal) --
    # the point is that it's no longer the previous router's real "sk-..." key.
    win._configure_panel.api_key_box.reveal_key(True)
    assert not win._configure_panel.api_key_box.key_label.text().startswith("sk-")
    assert win._configure_panel.api_key_box._api_key == ""
    assert "qwen" not in win._configure_panel.harness_box.harness_text.toPlainText()
    assert win._configure_panel.harness_box.harness_text.toPlainText() == ""


def test_report_validation_does_not_invent_router_errors(win):
    _member_profile(store.default_base_dir())
    _router_win(win)
    data = win._report.gather_report_data()
    joined = " ".join(data.get("validation", []))
    assert "at least one model" not in joined


def test_relaunch_does_not_force_kill_a_running_router(win, monkeypatch):
    _member_profile(store.default_base_dir())
    _router_win(win)
    monkeypatch.setattr(win._launch, "_validate_or_warn", lambda: True)
    monkeypatch.setattr("llama_launcher.ui.main_window.runtime.container_state",
                        lambda name, binary: "running")
    spawned = []
    monkeypatch.setattr(win._launch, "_spawn_async",
                        lambda argv, on_done=None: spawned.append(argv))
    # A live headless host may have a 100GB model resident and requests in
    # flight; Launch must not tear it down without asking.
    monkeypatch.setattr("llama_launcher.ui.controllers.launch_controller.QMessageBox.question",
                        staticmethod(lambda *a, **k: __import__(
                            "PySide6.QtWidgets", fromlist=["QMessageBox"]).QMessageBox.No))
    win._launch.on_launch()
    assert spawned == []


def test_router_load_failure_is_surfaced(win, monkeypatch):
    _router_win(win)
    monkeypatch.setattr("llama_launcher.ui.main_window.router_api.load_model",
                        lambda *a, **kw: False)
    win._monitor._on_router_load("qwen")
    assert "fail" in win._configure_panel.configure_status.status_label.text().lower()
    assert "fail" in win.monitor_status.status_label.text().lower()


def test_connected_state_distinguishes_down_from_empty(win, monkeypatch):
    _router_win(win)
    monkeypatch.setattr("llama_launcher.ui.main_window.router_api.list_models",
                        lambda host, port, key, **kw: None)
    win._monitor.refresh_router_models()
    assert "disconnected" in win._configure_panel.configure_status.status_label.text().lower()
    assert "disconnected" in win.monitor_status.status_label.text().lower()

    monkeypatch.setattr("llama_launcher.ui.main_window.router_api.list_models",
                        lambda host, port, key, **kw: [])
    win._monitor.refresh_router_models()
    # Reachable, just serving nothing -- not the same as unreachable.
    assert "disconnected" not in win._configure_panel.configure_status.status_label.text().lower()
    assert "disconnected" not in win.monitor_status.status_label.text().lower()


def test_failed_router_launch_is_reported(win):
    # Detached means no terminal: without this the only signal was a status
    # label stuck on "stopped". Called bare (as the router on_error callback
    # does), show_dialog defaults to False, so this can't pop a real modal
    # QMessageBox and hang the offscreen test run.
    win._launch._report_launch_error("Error: short-name resolution enforced but cannot prompt")
    assert "failed" in win.status_label.text().lower()
    assert "resolution" in win._configure_panel.configure_status.status_label.text()
    assert "resolution" in win.monitor_status.status_label.text()


def test_member_fields_are_editable(win):
    win._configure_panel.load_profile(Profile(name="Host", mode="router", image="img",
                             members=[RouterMember(profile="Qwen")]))
    # The spec's own recommended setup (load-on-startup for the primary model,
    # a pinned id matching what a harness already uses) must be reachable.
    win._configure_panel.set_member_fields(0, model_id="pinned-id", load_on_startup=True, stop_timeout=30)
    [m] = win._configure_panel.current_profile().members
    assert m.model_id == "pinned-id"
    assert m.load_on_startup is True
    assert m.stop_timeout == 30


def test_edited_member_fields_survive_a_save_load_round_trip(win):
    win._configure_panel.load_profile(Profile(name="Host", mode="router", image="img",
                             members=[RouterMember(profile="Qwen")]))
    win._configure_panel.set_member_fields(0, model_id="pinned", load_on_startup=True, stop_timeout=45)
    saved = win._configure_panel.current_profile()
    store.save_profile(saved, store.default_base_dir())
    back = store.load_profile(store.default_base_dir() / "profiles" / "host.json")
    assert back.members == [RouterMember(profile="Qwen", model_id="pinned",
                                         load_on_startup=True, stop_timeout=45)]


def test_member_whose_profile_is_gone_is_an_error_not_a_silent_drop(win):
    win._configure_panel.load_profile(Profile(name="Host", mode="router", image="img",
                             mounts=[Mount(host="/mnt/models", container="/models")],
                             members=[RouterMember(profile="Deleted Profile")],
                             settings={"port": 8080}))
    issues = win._configure_panel.router_issues()
    assert any("Deleted Profile" in i.message for i in issues if i.level == "error")


def test_spec_counters_feed_the_monitor_in_router_mode(win, monkeypatch):
    from llama_launcher.core.router_models import RouterModel
    _router_win(win, settings={"port": 8080, "metrics": True})
    monkeypatch.setattr("llama_launcher.ui.main_window.router_api.list_models",
                        lambda host, port, key, **kw: [RouterModel(id="qwen",
                                                                   status="loaded")])
    win._monitor.refresh_router_models()

    reads = iter([
        "llamacpp:spec_decode_num_draft_tokens_total 1000\n"
        "llamacpp:spec_decode_num_accepted_tokens_total 600\n"
        "llamacpp:spec_decode_num_drafts_total 300\n",
        "llamacpp:spec_decode_num_draft_tokens_total 2000\n"
        "llamacpp:spec_decode_num_accepted_tokens_total 1400\n"
        "llamacpp:spec_decode_num_drafts_total 700\n",
    ])
    seen = {}
    monkeypatch.setattr("llama_launcher.ui.main_window.metrics.fetch_metrics_text",
                        lambda port, **kw: seen.update(kw) or next(reads))
    p = win._configure_panel.current_profile()
    win._monitor._update_spec_stats(p)       # first read only primes the counters
    assert not win.monitor_panel.mtp_label.isVisibleTo(win.monitor_panel)
    win._monitor._update_spec_stats(p)
    text = win.monitor_panel.mtp_label.text()
    assert "80%" in text            # 800 accepted of 1000 drafted since last poll
    assert "counters" in text       # and it says where the number came from
    assert seen["model"] == "qwen"  # attributed to the model, not the router


def test_detached_checkbox_loads_from_profile(win):
    win._configure_panel.load_profile(Profile(name="Solo", image="img", model="/models/a.gguf",
                             runtime=Runtime(detached=True),
                             settings={"port": 8080}))
    assert win._configure_panel.detached_check.isChecked() is True


def test_detached_checkbox_saves_into_profile(win):
    win._configure_panel.load_profile(Profile(name="Solo", image="img", model="/models/a.gguf",
                             settings={"port": 8080}))
    assert win._configure_panel.current_profile().runtime.detached is False
    win._configure_panel.detached_check.setChecked(True)
    assert win._configure_panel.current_profile().runtime.detached is True


def test_detached_checkbox_hidden_in_router_mode(win):
    win._configure_panel.load_profile(Profile(name="Solo", image="img", model="/models/a.gguf",
                             settings={"port": 8080}))
    assert win._configure_panel.detached_check.isVisibleTo(win.centralWidget()) is True
    win._configure_panel.load_profile(Profile(name="Host", mode="router", image="img",
                             members=[RouterMember(profile="Qwen")]))
    assert win._configure_panel.detached_check.isVisibleTo(win.centralWidget()) is False


def test_member_candidates_include_profile_saved_this_session(win):
    # Regression: a member profile saved this session must appear in the
    # add-member list without an app restart. The natural flow leaves the new
    # model's name in the Name field while the form is back in router mode; the
    # old filter excluded exactly that name, hiding the just-made member.
    base = Path(win_base(win))
    _member_profile(base, name="modelB")
    win._configure_panel.name_edit.setText("modelB")                 # name still lingers in the field
    win._configure_panel.mode_combo.setCurrentIndex(win._configure_panel.mode_combo.findData("router"))
    assert "modelB" in win._configure_panel._member_candidates()


def test_member_candidates_exclude_routers(win):
    base = Path(win_base(win))
    _member_profile(base, name="modelA")
    store.save_profile(Profile(name="someRouter", mode="router", image="img"), base)
    cands = win._configure_panel._member_candidates()
    assert "modelA" in cands
    assert "someRouter" not in cands


def test_fresh_window_hides_router_widgets_in_default_server_mode(win):
    # Regression: on startup no profile is loaded and the mode combo defaults to
    # "server"; the router-only member widgets must be hidden without the user
    # having to flip the mode combo (which used to be the only trigger).
    assert win._configure_panel.mode_combo.currentData() == "server"
    assert win._configure_panel.add_member_btn.isVisibleTo(win.centralWidget()) is False
    assert win._configure_panel.members_list.isVisibleTo(win.centralWidget()) is False
    assert win._configure_panel.model_edit.isEnabled() is True


def test_load_mode_disables_legacy_mmap_widgets(win):
    # load-mode at default (mmap) leaves the legacy checkboxes usable...
    assert win._configure_panel._widgets["no-mmap"].isEnabled() is True
    assert win._configure_panel._widgets["mlock"].isEnabled() is True
    # ...but setting it to anything else grays them out (load-mode wins in argv).
    win._configure_panel._widgets["load-mode"].set_value("none")
    win._configure_panel._widgets["load-mode"].changed.emit()
    assert win._configure_panel._widgets["no-mmap"].isEnabled() is False
    assert win._configure_panel._widgets["mlock"].isEnabled() is False


def test_benchmark_has_its_own_tab_and_config_strip_hidden_off_configure(win):
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert titles == ["Configure", "Monitor", "Benchmark"]
    # Config-only strip (suggest-family + command preview) shows on Configure...
    win.tabs.setCurrentIndex(titles.index("Configure"))
    assert win._configure_panel._config_bottom.isVisibleTo(win) is True
    # ...and is hidden on Monitor/Benchmark.
    for name in ("Monitor", "Benchmark"):
        win.tabs.setCurrentIndex(titles.index(name))
        assert win._configure_panel._config_bottom.isVisibleTo(win) is False, name


def test_stopped_instance_card_offers_remove(win):
    # A stopped instance's card action button switches to Remove (podman rm),
    # carried over from the old table's ✕ button -- see StatCard.update_row.
    removed = []
    win.monitor_panel.instance_remove_requested.connect(removed.append)
    win.monitor_panel.set_instance_cards({"rows": [
        {"name": "llama-dead", "profile": "Dead", "port": 8080,
         "running": False, "health": "down", "stat": "", "tok_s": None, "kv_pct": None,
         "embeddings": False, "reranking": False, "mode": "server"}],
        "selected_name": None})
    win.monitor_panel.card("llama-dead").stop_button().click()
    assert removed == ["llama-dead"]


def test_running_instance_offers_stop(win):
    stopped = []
    win.monitor_panel.instance_stop_requested.connect(stopped.append)
    win.monitor_panel.set_instance_cards({"rows": [
        {"name": "llama-live", "profile": "Live", "port": 8080,
         "running": True, "health": "ready", "stat": "42 t/s", "tok_s": 42.0, "kv_pct": None,
         "embeddings": False, "reranking": False, "mode": "server"}],
        "selected_name": None})
    win.monitor_panel.card("llama-live").stop_button().click()
    assert stopped == ["llama-live"]


def test_router_status_clears_when_router_not_running(win, monkeypatch):
    # Regression: after a router stops/is removed, the models table + banners
    # kept showing its model list + "connected". update_status must clear
    # them when the container isn't running.
    win._configure_panel.load_profile(Profile(name="R", mode="router", image="img",
                             members=[RouterMember(profile="g12b")]))
    # seed a stale "connected + one model" state
    win._monitor._router_statuses = {"g12b": "loaded"}
    win.router_models_table.table.setRowCount(1)   # simulate a lingering model row
    win._set_router_connected(True)
    # container is absent (conftest default) -> update_status should clear
    win._monitor.update_status()
    assert win._monitor._router_statuses == {}
    assert win.router_models_table.table.rowCount() == 0
