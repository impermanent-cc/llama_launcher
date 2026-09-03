from llama_launcher.core.nodes import Node
from llama_launcher.core.spec import Mount, Profile, Runtime
from llama_launcher.services import runtime as runtime_svc
from llama_launcher.store.nodes import add_node
from llama_launcher.ui.controllers.launch_controller import LaunchController


def test_connection_resolves_from_profile_node(main_window, tmp_path):
    add_node(
        Node(name="box-b", kind="remote", connection="box-b", ssh_target="me@10.0.0.2"),
        main_window.base_dir(),
    )
    ctrl = main_window._launch
    local_p = Profile(name="a", runtime=Runtime(node="local"))
    remote_p = Profile(name="b", runtime=Runtime(node="box-b"))
    assert ctrl._connection_for_profile(local_p) == ""
    assert ctrl._connection_for_profile(remote_p) == "box-b"


def test_missing_node_falls_back_to_local(main_window):
    ctrl = main_window._launch
    p = Profile(name="c", runtime=Runtime(node="gone"))
    assert ctrl._connection_for_profile(p) == ""


def _remote_detached_profile():
    return Profile(
        name="d",
        image="img:tag",
        runtime=Runtime(node="box-b", detached=True),
        mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
        model="/models/m.gguf",
        settings={"port": 8080},
    )


def test_detached_launch_threads_connection_into_rm_argv_cleanup(
    main_window, monkeypatch
):
    """The pre-launch stale-container removal (podman rm -f) for a detached
    launch must target the SAME node as the run it's clearing the way for --
    otherwise a stale stopped container on the remote host is never cleaned up
    and the subsequent (correctly-targeted) remote run fails with "name
    already in use"."""
    add_node(
        Node(name="box-b", kind="remote", connection="box-b", ssh_target="me@10.0.0.2"),
        main_window.base_dir(),
    )
    monkeypatch.setattr(runtime_svc, "image_exists", lambda *a, **k: True)
    spawned = []
    monkeypatch.setattr(
        LaunchController,
        "_spawn_async",
        lambda self, argv, on_done=None, on_error=None: spawned.append(argv),
    )
    main_window._configure_panel.reload_nodes()
    main_window._configure_panel.load_profile(_remote_detached_profile())
    main_window._launch.on_launch()
    assert spawned, "expected the rm cleanup to be spawned"
    rm_argv = spawned[0]
    assert rm_argv[:3] == ["podman", "--connection", "box-b"]
    assert "rm" in rm_argv


def test_local_detached_launch_rm_argv_has_no_connection_flag(main_window, monkeypatch):
    """Local path stays byte-identical: no --connection flag on the cleanup rm."""
    monkeypatch.setattr(runtime_svc, "image_exists", lambda *a, **k: True)
    spawned = []
    monkeypatch.setattr(
        LaunchController,
        "_spawn_async",
        lambda self, argv, on_done=None, on_error=None: spawned.append(argv),
    )
    p = _remote_detached_profile()
    p.runtime.node = "local"
    main_window._configure_panel.load_profile(p)
    main_window._launch.on_launch()
    assert spawned, "expected the rm cleanup to be spawned"
    rm_argv = spawned[0]
    assert rm_argv == ["podman", "rm", "-f", "llama-d"]
