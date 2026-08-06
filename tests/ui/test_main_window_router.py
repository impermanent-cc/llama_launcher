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


def _member_profile(base, name="Qwen"):
    p = Profile(name=name, image="img", model="/models/qwen.gguf",
                mounts=[Mount(host="/mnt/models", container="/models")],
                settings={"ctx-size": 8192})
    store.save_profile(p, base)
    return p


def test_router_tab_exists(win):
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert "Router" in titles


def test_mode_round_trips_through_the_form(win):
    p = Profile(name="Host", mode="router", image="img",
                members=[RouterMember(profile="Qwen", model_id="qwen")])
    win.load_profile(p)
    back = win.current_profile()
    assert back.mode == "router"
    assert back.members == [RouterMember(profile="Qwen", model_id="qwen")]


def test_bind_host_round_trips(win):
    p = Profile(name="Host", mode="router", image="img",
                runtime=Runtime(bind_host="0.0.0.0"))
    win.load_profile(p)
    assert win.current_profile().runtime.bind_host == "0.0.0.0"


def test_prepare_router_files_writes_preset_and_key(win, tmp_path):
    base = store.default_base_dir()
    _member_profile(base)
    win.load_profile(Profile(name="Host", mode="router", image="img",
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
    win.load_profile(Profile(name="Host", mode="router", image="img",
                             mounts=[Mount(host="/mnt/models", container="/models")],
                             members=[RouterMember(profile="Qwen")],
                             settings={"port": 8080}))
    spawned = []
    monkeypatch.setattr(win, "_validate_or_warn", lambda: True)
    monkeypatch.setattr(win, "vram_check", lambda: None)

    pending = []

    def fake_spawn(argv, on_done=None, on_error=None):
        spawned.append(argv)
        # Do NOT call on_done inline: the real QProcess fires it later, so
        # calling it here would make a chained and an unchained implementation
        # produce identical output, which is exactly what this test must tell
        # apart.
        if on_done is not None:
            pending.append(on_done)

    monkeypatch.setattr(win, "_spawn_async", fake_spawn)
    win.on_launch()

    # Only the removal has been spawned so far; the run must be waiting on it.
    assert [a[:3] for a in spawned] == [["podman", "rm", "-f"]]
    assert pending, "the rm must carry a completion callback to chain the run"
    pending.pop(0)()            # rm finishes -> run starts

    run_argv = next(a for a in spawned if "run" in a)
    assert "-d" in run_argv
    assert "--models-preset" in run_argv
    assert ":/router:ro" in " ".join(run_argv)


def test_server_launch_still_uses_the_terminal(win, monkeypatch):
    win.load_profile(Profile(name="Solo", image="img", model="/models/a.gguf",
                             mounts=[Mount(host="/h", container="/models")],
                             settings={"port": 8080}))
    called = {}
    monkeypatch.setattr(win, "_validate_or_warn", lambda: True)
    monkeypatch.setattr(win, "vram_check", lambda: None)
    monkeypatch.setattr("llama_launcher.ui.main_window.terminal.launch",
                        lambda argv: called.setdefault("argv", argv))
    win.on_launch()
    assert "-d" not in called["argv"]


def test_adopt_running_containers_reads_the_label_list(win, monkeypatch):
    monkeypatch.setattr(
        "llama_launcher.ui.main_window.runtime.list_launcher_containers",
        lambda binary: [{"name": "llama-host", "running": True,
                         "profile": "Host", "mode": "router"}])
    rows = win.adopt_running_containers()
    assert rows[0]["profile"] == "Host"


def test_refresh_router_models_populates_the_panel(win, monkeypatch):
    from llama_launcher.core.router_models import RouterModel
    win.load_profile(Profile(name="Host", mode="router", image="img",
                             settings={"port": 8080}))
    monkeypatch.setattr(
        "llama_launcher.ui.main_window.router_api.list_models",
        lambda host, port, key, **kw: [RouterModel(id="qwen", status="sleeping")])
    win.refresh_router_models()
    assert win.router_panel.table.rowCount() == 1
    assert win.router_panel.table.item(0, 0).text() == "qwen"


def test_sleeping_models_are_not_polled_for_metrics(win, monkeypatch):
    """A sleeping model has nothing to measure, and skipping it is the point of
    an idle-unloading host (plan self-review gap 2)."""
    from llama_launcher.core.router_models import RouterModel
    win.load_profile(Profile(name="Host", mode="router", image="img",
                             settings={"port": 8080, "metrics": True}))
    monkeypatch.setattr(
        "llama_launcher.ui.main_window.router_api.list_models",
        lambda host, port, key, **kw: [RouterModel(id="qwen", status="sleeping")])
    win.refresh_router_models()

    calls = []
    monkeypatch.setattr("llama_launcher.ui.main_window.metrics.fetch_metrics",
                        lambda *a, **kw: calls.append(kw) or {})
    monkeypatch.setattr("llama_launcher.ui.main_window.metrics.fetch_slots",
                        lambda *a, **kw: calls.append(kw) or [])
    win.collect_monitor_data()
    assert calls == []


def test_loaded_model_is_polled_with_its_id(win, monkeypatch):
    from llama_launcher.core.router_models import RouterModel
    win.load_profile(Profile(name="Host", mode="router", image="img",
                             settings={"port": 8080, "metrics": True}))
    monkeypatch.setattr(
        "llama_launcher.ui.main_window.router_api.list_models",
        lambda host, port, key, **kw: [RouterModel(id="qwen", status="loaded")])
    win.refresh_router_models()

    seen = {}
    monkeypatch.setattr("llama_launcher.ui.main_window.metrics.fetch_metrics",
                        lambda port, **kw: seen.update(kw) or {})
    monkeypatch.setattr("llama_launcher.ui.main_window.metrics.fetch_slots",
                        lambda port, **kw: [])
    win.collect_monitor_data()
    assert seen["model"] == "qwen"


def test_router_form_hides_model_level_settings(win):
    win.load_profile(Profile(name="Host", mode="router", image="img"))
    # The precedence trap: router CLI args outrank every member's preset value.
    assert not win._widgets["ctx-size"].isVisibleTo(win.configure_tab)
    assert win._widgets["models-max"].isVisibleTo(win.configure_tab)


def test_server_form_hides_router_only_settings(win):
    win.load_profile(Profile(name="Solo", image="img", model="/m.gguf"))
    assert not win._widgets["models-max"].isVisibleTo(win.configure_tab)
    assert win._widgets["ctx-size"].isVisibleTo(win.configure_tab)


def test_server_profile_cannot_emit_router_only_flags(win):
    from llama_launcher.core.command_builder import build_command
    win.load_profile(Profile(name="Solo", image="img", model="/m.gguf",
                             mounts=[Mount(host="/h", container="/models")],
                             settings={"port": 8080, "models-max": 2}))
    # models-max is router-only; a single-model llama-server rejects it.
    assert "--models-max" not in build_command(win.current_profile())


def _router_win(win, **kw):
    base = dict(name="Host", mode="router", image="img",
                mounts=[Mount(host="/mnt/models", container="/models")],
                members=[RouterMember(profile="Qwen")], settings={"port": 8080})
    base.update(kw)
    win.load_profile(Profile(**base))
    return win


def test_preview_includes_the_router_mount(win):
    _member_profile(store.default_base_dir())
    _router_win(win)
    assert ":/router:ro" in win.preview_text()


def test_exported_script_includes_the_router_mount(win, tmp_path):
    _member_profile(store.default_base_dir())
    _router_win(win)
    out = tmp_path / "run.sh"
    win.export_sh(str(out))
    text = out.read_text()
    assert ":/router:ro" in text
    # The preset and key paths are only reachable because of that mount.
    assert "--models-preset" in text


def test_router_tab_populates_on_profile_load_not_only_on_launch(win):
    _member_profile(store.default_base_dir())
    _router_win(win)
    # Reattach-after-restart is the common path: select the profile, see the key.
    assert "qwen" in win.router_panel.harness_text.toPlainText()
    win.router_panel.reveal_key(True)
    assert win.router_panel.key_label.text().startswith("sk-")


def test_exposure_banner_shows_on_load_for_a_non_loopback_router(win):
    _member_profile(store.default_base_dir())
    _router_win(win, runtime=Runtime(bind_host="0.0.0.0"))
    assert "0.0.0.0" in win.router_panel.banner.text()


def test_report_validation_does_not_invent_router_errors(win):
    _member_profile(store.default_base_dir())
    _router_win(win)
    data = win.gather_report_data()
    joined = " ".join(data.get("validation", []))
    assert "at least one model" not in joined


def test_relaunch_does_not_force_kill_a_running_router(win, monkeypatch):
    _member_profile(store.default_base_dir())
    _router_win(win)
    monkeypatch.setattr(win, "_validate_or_warn", lambda: True)
    monkeypatch.setattr("llama_launcher.ui.main_window.runtime.container_state",
                        lambda name, binary: "running")
    spawned = []
    monkeypatch.setattr(win, "_spawn_async",
                        lambda argv, on_done=None: spawned.append(argv))
    # A live headless host may have a 100GB model resident and requests in
    # flight; Launch must not tear it down without asking.
    monkeypatch.setattr("llama_launcher.ui.main_window.QMessageBox.question",
                        staticmethod(lambda *a, **k: __import__(
                            "PySide6.QtWidgets", fromlist=["QMessageBox"]).QMessageBox.No))
    win.on_launch()
    assert spawned == []


def test_router_load_failure_is_surfaced(win, monkeypatch):
    _router_win(win)
    monkeypatch.setattr("llama_launcher.ui.main_window.router_api.load_model",
                        lambda *a, **kw: False)
    win._on_router_load("qwen")
    assert "fail" in win.router_panel.status_label.text().lower()


def test_connected_state_distinguishes_down_from_empty(win, monkeypatch):
    _router_win(win)
    monkeypatch.setattr("llama_launcher.ui.main_window.router_api.list_models",
                        lambda host, port, key, **kw: None)
    win.refresh_router_models()
    assert "disconnected" in win.router_panel.status_label.text().lower()

    monkeypatch.setattr("llama_launcher.ui.main_window.router_api.list_models",
                        lambda host, port, key, **kw: [])
    win.refresh_router_models()
    # Reachable, just serving nothing -- not the same as unreachable.
    assert "disconnected" not in win.router_panel.status_label.text().lower()


def test_failed_router_launch_is_reported(win):
    # Detached means no terminal: without this the only signal was a status
    # label stuck on "stopped".
    win._report_launch_error("Error: short-name resolution enforced but cannot prompt")
    assert "failed" in win.status_label.text().lower()
    assert "resolution" in win.router_panel.status_label.text()


def test_member_fields_are_editable(win):
    win.load_profile(Profile(name="Host", mode="router", image="img",
                             members=[RouterMember(profile="Qwen")]))
    # The spec's own recommended setup (load-on-startup for the primary model,
    # a pinned id matching what a harness already uses) must be reachable.
    win.set_member_fields(0, model_id="pinned-id", load_on_startup=True, stop_timeout=30)
    [m] = win.current_profile().members
    assert m.model_id == "pinned-id"
    assert m.load_on_startup is True
    assert m.stop_timeout == 30


def test_edited_member_fields_survive_a_save_load_round_trip(win):
    win.load_profile(Profile(name="Host", mode="router", image="img",
                             members=[RouterMember(profile="Qwen")]))
    win.set_member_fields(0, model_id="pinned", load_on_startup=True, stop_timeout=45)
    saved = win.current_profile()
    store.save_profile(saved, store.default_base_dir())
    back = store.load_profile(store.default_base_dir() / "profiles" / "host.json")
    assert back.members == [RouterMember(profile="Qwen", model_id="pinned",
                                         load_on_startup=True, stop_timeout=45)]


def test_member_whose_profile_is_gone_is_an_error_not_a_silent_drop(win):
    win.load_profile(Profile(name="Host", mode="router", image="img",
                             mounts=[Mount(host="/mnt/models", container="/models")],
                             members=[RouterMember(profile="Deleted Profile")],
                             settings={"port": 8080}))
    issues = win.router_issues()
    assert any("Deleted Profile" in i.message for i in issues if i.level == "error")


def test_spec_counters_feed_the_monitor_in_router_mode(win, monkeypatch):
    from llama_launcher.core.router_models import RouterModel
    _router_win(win, settings={"port": 8080, "metrics": True})
    monkeypatch.setattr("llama_launcher.ui.main_window.router_api.list_models",
                        lambda host, port, key, **kw: [RouterModel(id="qwen",
                                                                   status="loaded")])
    win.refresh_router_models()

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
    p = win.current_profile()
    win._update_spec_stats(p)       # first read only primes the counters
    assert not win.monitor_panel.mtp_label.isVisibleTo(win.monitor_panel)
    win._update_spec_stats(p)
    text = win.monitor_panel.mtp_label.text()
    assert "80%" in text            # 800 accepted of 1000 drafted since last poll
    assert "counters" in text       # and it says where the number came from
    assert seen["model"] == "qwen"  # attributed to the model, not the router
