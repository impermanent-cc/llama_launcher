from llama_launcher.core.spec import Profile, Runtime, Mount, RpcWorker
from llama_launcher.core.command_builder import build_command, build_worker_command


def _p(binary="podman"):
    return Profile(name="p", image="ghcr.io/ggml-org/llama.cpp:server-cuda13",
                   runtime=Runtime(binary=binary),
                   model="/models/m.gguf", settings={"port": 8080})


def test_local_connection_is_byte_for_byte_unchanged():
    p = _p()
    assert build_command(p, connection="") == build_command(p)


def test_remote_connection_inserted_after_binary_before_run():
    argv = build_command(_p(), connection="box-b")
    assert argv[0] == "podman"
    assert argv[1:3] == ["--connection", "box-b"]
    assert argv[3] == "run"
    assert "--connection" not in build_command(_p())     # default still clean


def test_remote_docker_node_uses_context_not_connection():
    # docker selects a remote host with --context; --connection is podman-only
    # and docker rejects it ("unknown flag"). runtime._base already branches this
    # way for stop/status -- the launch argv must match or remote-docker launch
    # fails while stop/status work (an inconsistency).
    argv = build_command(_p("docker"), connection="box-b")
    assert argv[0] == "docker"
    assert argv[1:3] == ["--context", "box-b"]
    assert "--connection" not in argv
    assert argv[3] == "run"


def test_remote_docker_worker_uses_context_not_connection():
    argv = build_worker_command(_p("docker"), RpcWorker(node="box-b", device="CPU"),
                                index=0, connection="box-b")
    assert argv[0] == "docker"
    assert argv[1:3] == ["--context", "box-b"]
    assert "--connection" not in argv
