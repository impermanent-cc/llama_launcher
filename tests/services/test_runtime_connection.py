from unittest.mock import patch
from llama_launcher.services import runtime


def test_stop_and_rm_and_logs_prefix_connection():
    assert runtime.stop_argv("llama-p", "podman", 10, connection="box-b") == \
        ["podman", "--connection", "box-b", "stop", "-t", "10", "llama-p"]
    assert runtime.rm_argv("llama-p", "podman", connection="box-b") == \
        ["podman", "--connection", "box-b", "rm", "-f", "llama-p"]
    assert runtime.logs_argv("llama-p", "podman", connection="box-b") == \
        ["podman", "--connection", "box-b", "logs", "-f", "llama-p"]


def test_local_argv_unchanged():
    assert runtime.stop_argv("llama-p", "podman", 10) == \
        ["podman", "stop", "-t", "10", "llama-p"]


def test_pull_and_connection_mgmt_argv():
    assert runtime.pull_argv("img:tag", "podman", connection="box-b") == \
        ["podman", "--connection", "box-b", "pull", "img:tag"]
    assert runtime.connection_add_argv("box-b", "me@10.0.0.2:22") == \
        ["podman", "system", "connection", "add", "box-b", "ssh://me@10.0.0.2:22"]
    assert runtime.connection_remove_argv("box-b") == \
        ["podman", "system", "connection", "remove", "box-b"]


def test_list_launcher_containers_passes_connection_to_run():
    seen = {}
    def fake_run(args, timeout=10):
        seen["args"] = args
        class R: returncode = 0; stdout = "[]"
        return R()
    with patch.object(runtime, "_run", fake_run):
        runtime.list_launcher_containers("podman", connection="box-b")
    assert seen["args"][:3] == ["podman", "--connection", "box-b"]
    assert "ps" in seen["args"]


def test_image_exists_and_node_reachable_use_returncode():
    def rc0(args, timeout=10):
        class R: returncode = 0; stdout = ""
        return R()
    with patch.object(runtime, "_run", rc0):
        assert runtime.image_exists("img", "podman", connection="box-b") is True
        assert runtime.node_reachable("box-b", "podman") is True


def test_container_exists_and_started_at_prefix_connection():
    """The focused Monitor / log-follower path needs these two threaded with
    --connection too, so a remote instance's existence/uptime checks hit the
    right node's podman instead of local."""
    seen = {}
    def fake_run(args, timeout=10):
        seen["args"] = args
        class R: returncode = 0; stdout = "2024-01-15T10:30:00Z\n"
        return R()
    with patch.object(runtime, "_run", fake_run):
        assert runtime.container_exists("llama-p", "podman", connection="box-b") is True
        assert seen["args"] == ["podman", "--connection", "box-b", "container", "exists", "llama-p"]

        assert runtime.started_at("llama-p", "podman", connection="box-b") == "2024-01-15T10:30:00Z"
        assert seen["args"] == ["podman", "--connection", "box-b", "inspect", "-f",
                                "{{.State.StartedAt}}", "llama-p"]


def test_container_exists_and_started_at_local_argv_unchanged():
    seen = {}
    def fake_run(args, timeout=10):
        seen["args"] = args
        class R: returncode = 0; stdout = "2024-01-15T10:30:00Z\n"
        return R()
    with patch.object(runtime, "_run", fake_run):
        runtime.container_exists("llama-p", "podman")
        assert seen["args"] == ["podman", "container", "exists", "llama-p"]

        runtime.started_at("llama-p", "podman")
        assert seen["args"] == ["podman", "inspect", "-f", "{{.State.StartedAt}}", "llama-p"]
