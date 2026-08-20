from llama_launcher.services import gpu


def test_local_argv_is_plain_nvidia_smi():
    argv = gpu.nvidia_smi_argv()
    assert argv[0] == "nvidia-smi"
    assert any(a.startswith("--query-gpu=") for a in argv)


def test_remote_argv_wraps_in_ssh():
    argv = gpu.nvidia_smi_argv("me@10.0.0.2")
    assert argv[:3] == ["ssh", "me@10.0.0.2", "nvidia-smi"]
    assert any(a.startswith("--query-gpu=") for a in argv)
