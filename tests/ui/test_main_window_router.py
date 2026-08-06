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

    def fake_spawn(argv, on_done=None):
        spawned.append(argv)
        if on_done is not None:
            on_done()           # the real QProcess fires this when it exits

    monkeypatch.setattr(win, "_spawn_async", fake_spawn)
    win.on_launch()
    # The stale container is removed first, then the run is chained off it.
    assert spawned[0][:3] == ["podman", "rm", "-f"]
    run_argv = next(a for a in spawned if "run" in a)
    assert "-d" in run_argv
    assert "--models-preset" in run_argv


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
