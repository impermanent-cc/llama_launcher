from llama_launcher.core.spec import Profile, Mount, Runtime
from llama_launcher.core.command_builder import build_command


def _base_profile():
    return Profile(
        name="My Model",
        image="ghcr.io/ggml-org/llama.cpp:server-cuda12-b9628",
        runtime=Runtime(binary="podman", gpu_mode="cdi"),
        mounts=[Mount(host="/mnt/storage/AI/Models", container="/models",
                      role="model", mode="ro")],
        model="/models/m.gguf",
        settings={"port": 8080},
    )


def test_run_prefix_cdi():
    argv = build_command(_base_profile())
    # Membership rather than position: reattach labels are inserted between
    # --name and --device, and the builder's ordering is not part of the contract.
    assert argv[:3] == ["podman", "run", "--rm"]
    assert argv[argv.index("--name") + 1] == "llama-my-model"
    assert "--device" in argv
    assert "nvidia.com/gpu=all" in argv
    assert "-p" in argv and "127.0.0.1:8080:8080" in argv
    assert "-v" in argv and "/mnt/storage/AI/Models:/models:ro" in argv
    # image appears before the model arg
    img = "ghcr.io/ggml-org/llama.cpp:server-cuda12-b9628"
    assert img in argv
    assert argv.index(img) < argv.index("-m")


def test_gpus_all_mode_and_selinux_label():
    p = _base_profile()
    p.runtime.gpu_mode = "gpus-all"
    p.runtime.selinux_label_disable = True
    argv = build_command(p)
    assert "--gpus" in argv and "all" in argv
    assert "--security-opt=label=disable" in argv
    assert "--device" not in argv


def test_gpu_none_mode():
    p = _base_profile()
    p.runtime.gpu_mode = "none"
    argv = build_command(p)
    assert "--device" not in argv
    assert "--gpus" not in argv


def test_workspace_workdir_and_rw_and_selinux_opt():
    p = _base_profile()
    p.mounts.append(Mount(host="/home/me/ws", container="/workspace",
                          role="workspace", mode="rw", selinux="z", workdir=True))
    argv = build_command(p)
    assert "/home/me/ws:/workspace:rw,z" in argv
    assert "-w" in argv and "/workspace" in argv


def test_blank_mount_is_skipped():
    p = _base_profile()
    p.mounts.append(Mount(host="", container="", role="custom", mode="ro"))
    argv = build_command(p)
    # only the one real mount should produce a -v
    assert argv.count("-v") == 1


def test_docker_binary_and_extra_run_args():
    p = _base_profile()
    p.runtime.binary = "docker"
    p.runtime.extra_run_args = "--cpus 4"
    argv = build_command(p)
    assert argv[0] == "docker"
    assert "--cpus" in argv and "4" in argv
