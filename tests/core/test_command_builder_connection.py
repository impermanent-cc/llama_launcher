from llama_launcher.core.spec import Profile, Runtime, Mount
from llama_launcher.core.command_builder import build_command


def _p():
    return Profile(name="p", image="ghcr.io/ggml-org/llama.cpp:server-cuda13",
                   runtime=Runtime(binary="podman"),
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
