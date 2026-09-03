import pytest

import llama_launcher.ui.main_window as mw
from llama_launcher.core.spec import Mount, Profile, Runtime
from llama_launcher.ui.controllers import launch_controller
from llama_launcher.ui.controllers.monitor_controller import MonitorController


@pytest.fixture(autouse=True)
def _stub_dialogs(monkeypatch):
    # Modal dialogs would block forever in the headless/offscreen test run.
    monkeypatch.setattr(launch_controller.QMessageBox, "critical", lambda *a, **k: None)
    monkeypatch.setattr(launch_controller.QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(
        launch_controller.QMessageBox, "information", lambda *a, **k: None
    )


def _profile():
    return Profile(
        name="Act",
        image="img:tag",
        runtime=Runtime(binary="podman", gpu_mode="cdi"),
        mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
        model="/models/m.gguf",
        settings={"port": 8080},
    )


def test_save_and_reload_profile(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(mw, "base_dir", lambda: tmp_path)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(_profile())
    w.save_current_profile()
    # new window sees the saved profile in its dropdown
    w2 = mw.MainWindow()
    qtbot.addWidget(w2)
    names = [
        w2._configure_panel.profile_combo.itemText(i)
        for i in range(w2._configure_panel.profile_combo.count())
    ]
    assert "Act" in names


def test_startup_does_not_show_first_profile_as_loaded(qtbot, tmp_path, monkeypatch):
    """At startup the dropdown must not display a saved profile as if loaded:
    addItems auto-selects index 0 without running load_profile, which would
    leave the form on blank defaults under a real profile name. The combo
    starts unselected (a 'choose a profile' placeholder) until the user picks."""
    monkeypatch.setattr(mw, "base_dir", lambda: tmp_path)
    seed = mw.MainWindow()
    qtbot.addWidget(seed)
    seed._configure_panel.load_profile(_profile())  # name="Act"
    seed.save_current_profile()

    w = mw.MainWindow()  # fresh startup
    qtbot.addWidget(w)
    combo = w._configure_panel.profile_combo
    assert combo.count() >= 1  # the saved profile IS listed
    assert combo.currentIndex() == -1  # ...but nothing is selected
    assert combo.placeholderText()  # a hint is shown instead
    # the form is still the blank default, matching the un-selected combo
    assert w._configure_panel._profile.name == "New Profile"


def test_name_field_is_saved(qtbot, tmp_path, monkeypatch):
    """A name typed into the Name field is the name the profile saves under."""
    monkeypatch.setattr(mw, "base_dir", lambda: tmp_path)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(_profile())  # name="Act"
    w._configure_panel.name_edit.setText("Renamed Build")
    w.save_current_profile()
    w2 = mw.MainWindow()
    qtbot.addWidget(w2)
    names = [
        w2._configure_panel.profile_combo.itemText(i)
        for i in range(w2._configure_panel.profile_combo.count())
    ]
    assert "Renamed Build" in names


def test_on_launch_does_not_follow_logs_before_container_exists(qtbot, monkeypatch):
    """The container is created asynchronously by the launched terminal, so right
    after on_launch() it does not exist yet. Attaching `podman logs -f` then just
    captures 'no such container' and dies, leaving the logs pane stuck on it. So
    on_launch must NOT start a follower against a not-yet-existing container."""
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    monkeypatch.setattr(mw.terminal, "launch", lambda *a, **k: None)
    monkeypatch.setattr(
        mw.runtime, "container_exists", lambda name, binary, connection="": False
    )
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(_profile())
    w._launch.on_launch()
    assert w._monitor._log_proc is None


def test_start_log_follower_skips_when_container_absent(qtbot, monkeypatch):
    monkeypatch.setattr(
        mw.runtime, "container_exists", lambda name, binary, connection="": False
    )
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._monitor._start_log_follower()
    assert w._monitor._log_proc is None


def test_start_log_follower_threads_remote_node_connection(
    qtbot, tmp_path, monkeypatch
):
    """The log-follower for a focused REMOTE instance must query
    container_exists AND build its logs argv against that node's podman
    --connection, never local podman -- otherwise it checks LOCAL podman
    (which never has the remote container) and the logs pane for a remote
    server never populates."""
    from llama_launcher.core.instances import Instance
    from llama_launcher.core.nodes import Node
    from llama_launcher.core.spec import Profile
    from llama_launcher.store.nodes import add_node

    monkeypatch.setattr(mw, "base_dir", lambda: tmp_path)
    add_node(
        Node(name="box-b", kind="remote", connection="box-b", ssh_target="me@10.0.0.2"),
        tmp_path,
    )

    exists_calls = []
    argv_calls = []
    monkeypatch.setattr(
        mw.runtime,
        "container_exists",
        lambda name, binary, connection="": exists_calls.append(connection) or True,
    )
    monkeypatch.setattr(
        mw.runtime,
        "logs_argv",
        lambda name, binary, connection="": (
            argv_calls.append(connection)
            or ["podman", "--connection", connection, "logs", "-f", name]
        ),
    )

    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(
        Profile(name="Solo", image="img", settings={"port": 8080})
    )
    w._monitor._instances = [
        Instance(
            name="llama-rem",
            profile="rem",
            mode="server",
            running=True,
            port=8081,
            host="10.0.0.2",
            embeddings=False,
            reranking=False,
            stop_timeout=10,
            binary="podman",
            node="box-b",
        )
    ]
    w._monitor._active_instance = w._monitor._instances[0]

    w._monitor._start_log_follower()

    assert exists_calls == ["box-b"]
    assert argv_calls == ["box-b"]


def test_update_status_starts_follower_when_running(qtbot, monkeypatch):
    """Once the container is actually running and no follower is attached, the
    status poll starts one (logs replay from the start, so nothing is missed)."""
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    monkeypatch.setattr(
        mw.runtime, "container_state", lambda name, binary, connection="": "running"
    )
    monkeypatch.setattr(mw.health, "probe_health", lambda port, **kw: "ready")
    monkeypatch.setattr(MonitorController, "_log_follower_active", lambda self: False)
    calls = []
    monkeypatch.setattr(
        MonitorController, "_start_log_follower", lambda self: calls.append(1)
    )
    w = mw.MainWindow()
    qtbot.addWidget(w)
    calls.clear()
    w._monitor.update_status()
    assert calls == [1]


def test_update_status_does_not_restart_active_follower(qtbot, monkeypatch):
    """A follower already streaming must not be re-spawned on every poll."""
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    monkeypatch.setattr(
        mw.runtime, "container_state", lambda name, binary, connection="": "running"
    )
    monkeypatch.setattr(mw.health, "probe_health", lambda port, **kw: "ready")
    monkeypatch.setattr(MonitorController, "_log_follower_active", lambda self: True)
    calls = []
    monkeypatch.setattr(
        MonitorController, "_start_log_follower", lambda self: calls.append(1)
    )
    w = mw.MainWindow()
    qtbot.addWidget(w)
    calls.clear()
    w._monitor.update_status()
    assert calls == []


def test_on_launch_invokes_terminal(qtbot, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        mw.terminal,
        "launch",
        lambda argv, template=mw.terminal.DEFAULT_TEMPLATE: captured.setdefault(
            "argv", argv
        ),
    )
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(_profile())
    w._launch.on_launch()
    assert captured["argv"][0] == "podman"
    assert "img:tag" in captured["argv"]


def test_on_launch_blocks_on_validation_error(qtbot, monkeypatch):
    called = {"launched": False}
    monkeypatch.setattr(
        mw.terminal, "launch", lambda *a, **k: called.__setitem__("launched", True)
    )
    monkeypatch.setattr(mw.runtime, "binary_available", lambda b: True)
    w = mw.MainWindow()
    qtbot.addWidget(w)
    p = _profile()
    p.model = ""  # invalid
    w._configure_panel.load_profile(p)
    w._launch.on_launch()
    assert called["launched"] is False


def test_fetch_latest_updates_image(qtbot, monkeypatch):
    # on_fetch_latest runs the lookup on a worker thread; stub
    # _UpdateWorker.start to emit `found` synchronously rather than spinning a
    # real QThread, and stub the resulting info dialog to avoid a modal exec()
    # in headless tests.
    monkeypatch.setattr(
        mw.registry,
        "fetch_latest",
        lambda repo, prefix, timeout=10.0: "server-cuda12-b9999",
    )
    monkeypatch.setattr(
        launch_controller.QMessageBox, "information", lambda *a, **k: None
    )
    monkeypatch.setattr(
        launch_controller._UpdateWorker,
        "start",
        lambda self: self.found.emit("server-cuda12-b9999"),
    )
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.image_edit.setText("ghcr.io/ggml-org/llama.cpp:server-cuda12-b1")
    w._launch.on_fetch_latest()
    assert (
        w._configure_panel.image_edit.text()
        == "ghcr.io/ggml-org/llama.cpp:server-cuda12-b9999"
    )


def test_quit_app_stops_status_timer(qtbot):
    w = mw.MainWindow()
    qtbot.addWidget(w)
    assert w._status_timer.isActive()  # running after construction
    w.quit_app()
    assert not w._status_timer.isActive()  # stopped on real teardown


def test_close_no_tray_stops_status_timer(qtbot):
    w = mw.MainWindow()
    qtbot.addWidget(w)
    w._minimize_to_tray = False  # default; close() takes the real-quit branch
    assert w._status_timer.isActive()
    w.close()
    assert not w._status_timer.isActive()
