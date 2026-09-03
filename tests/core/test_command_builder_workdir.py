from llama_launcher.core.command_builder import build_command
from llama_launcher.core.spec import Mount, Profile, Runtime


def _profile(workdir):
    return Profile(
        name="p",
        image="img:tag",
        runtime=Runtime(binary="podman", gpu_mode="cdi"),
        mounts=[
            Mount(host="/h/models", container="/models", role="model", mode="ro"),
            Mount(
                host="/h/ws",
                container="/workspace",
                role="workspace",
                mode="rw",
                workdir=workdir,
            ),
        ],
        model="/models/m.gguf",
        settings={"port": 8080},
    )


def test_workdir_pins_ld_library_path():
    argv = build_command(_profile(workdir=True))
    assert argv[argv.index("-w") + 1] == "/workspace"
    # LD_LIBRARY_PATH=/app must accompany the workdir so the image's libs load
    assert "-e" in argv
    assert "LD_LIBRARY_PATH=/app" in argv
    # it's a podman run option, so it must precede the image
    assert argv.index("LD_LIBRARY_PATH=/app") < argv.index("img:tag")


def test_no_workdir_means_no_ld_library_path():
    argv = build_command(_profile(workdir=False))
    assert "-w" not in argv
    assert "LD_LIBRARY_PATH=/app" not in argv
