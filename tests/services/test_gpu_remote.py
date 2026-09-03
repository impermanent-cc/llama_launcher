import pytest

from llama_launcher.services import gpu


def test_local_argv_is_plain_nvidia_smi():
    argv = gpu.nvidia_smi_argv()
    assert argv[0] == "nvidia-smi"
    assert any(a.startswith("--query-gpu=") for a in argv)


def test_remote_argv_wraps_in_ssh():
    argv = gpu.nvidia_smi_argv("me@10.0.0.2")
    assert argv[0] == "ssh"
    assert "-o" in argv and "BatchMode=yes" in argv  # no hang on host-key prompt
    assert "me@10.0.0.2" in argv
    assert "nvidia-smi" in argv
    # the ssh options precede the target, the target precedes the command
    assert argv.index("me@10.0.0.2") < argv.index("nvidia-smi")
    assert any(a.startswith("--query-gpu=") for a in argv)


def test_nvidia_smi_argv_rejects_unsafe_proxycommand():
    with pytest.raises(ValueError, match="unsafe ssh target"):
        gpu.nvidia_smi_argv("-oProxyCommand=x")


def test_nvidia_smi_argv_rejects_leading_dash():
    with pytest.raises(ValueError, match="unsafe ssh target"):
        gpu.nvidia_smi_argv("-x")


def test_query_gpus_returns_empty_for_unsafe_target():
    # query_gpus must NOT crash; it catches ValueError and returns []
    result = gpu.query_gpus("-oProxyCommand=x")
    assert result == []
