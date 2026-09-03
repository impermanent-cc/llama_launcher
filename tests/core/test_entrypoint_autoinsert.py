from llama_launcher.core.command_builder import (
    build_command,
    image_tag,
    needs_server_entrypoint,
)
from llama_launcher.core.spec import Mount, Profile, Runtime


def test_image_tag():
    assert image_tag("ghcr.io/ggml-org/llama.cpp:full") == "full"
    assert (
        image_tag("ghcr.io/ggml-org/llama.cpp:server-cuda12-b9628")
        == "server-cuda12-b9628"
    )
    assert image_tag("ghcr.io/ggml-org/llama.cpp") == ""  # no tag
    assert image_tag("localhost:5000/llama.cpp") == ""  # host:port, no tag


def test_needs_server_entrypoint():
    assert needs_server_entrypoint("ghcr.io/ggml-org/llama.cpp:full") is True
    assert needs_server_entrypoint("ghcr.io/ggml-org/llama.cpp:full-cuda") is True
    assert needs_server_entrypoint("ghcr.io/ggml-org/llama.cpp:light") is True
    assert needs_server_entrypoint("ghcr.io/ggml-org/llama.cpp:server") is False
    assert (
        needs_server_entrypoint("ghcr.io/ggml-org/llama.cpp:server-cuda12-b9628")
        is False
    )
    assert needs_server_entrypoint("ghcr.io/ggml-org/llama.cpp") is False


def _profile(image, extra=""):
    return Profile(
        name="p",
        image=image,
        runtime=Runtime(binary="podman", extra_run_args=extra),
        mounts=[Mount(host="/h", container="/Models", role="model", mode="ro")],
        model="/Models/m.gguf",
        settings={"port": 8080},
    )


def test_full_image_autoinjects_entrypoint():
    argv = build_command(_profile("ghcr.io/ggml-org/llama.cpp:full"))
    assert "--entrypoint" in argv
    i = argv.index("--entrypoint")
    assert argv[i + 1] == "/app/llama-server"
    assert i < argv.index("ghcr.io/ggml-org/llama.cpp:full")  # before the image
    assert "-m" in argv  # server flag still emitted


def test_server_image_no_entrypoint():
    argv = build_command(_profile("ghcr.io/ggml-org/llama.cpp:server-cuda12-b9628"))
    assert "--entrypoint" not in argv


def test_user_entrypoint_not_overridden():
    argv = build_command(
        _profile("ghcr.io/ggml-org/llama.cpp:full", extra="--entrypoint /custom/bin")
    )
    assert argv.count("--entrypoint") == 1  # not doubled
    i = argv.index("--entrypoint")
    assert argv[i + 1] == "/custom/bin"
