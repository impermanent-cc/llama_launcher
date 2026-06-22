from llama_launcher.core.spec import Mount
from llama_launcher.core.pathmap import host_to_container


def test_file_under_mount_maps():
    mounts = [Mount(host="/h/models", container="/models")]
    assert host_to_container("/h/models/q4.gguf", mounts) == "/models/q4.gguf"


def test_file_under_no_mount_returns_none():
    mounts = [Mount(host="/h/models", container="/models")]
    assert host_to_container("/other/file.gguf", mounts) is None


def test_nested_subdirs_map():
    mounts = [Mount(host="/h/models", container="/models")]
    assert host_to_container("/h/models/a/b/c.gguf", mounts) == "/models/a/b/c.gguf"


def test_exact_match_returns_container():
    mounts = [Mount(host="/h/models", container="/models")]
    assert host_to_container("/h/models", mounts) == "/models"


def test_trailing_slash_hosts_handled():
    mounts = [Mount(host="/h/models/", container="/models/")]
    assert host_to_container("/h/models/q4.gguf", mounts) == "/models/q4.gguf"


def test_skips_empty_mounts():
    mounts = [Mount(host="", container="/models"),
              Mount(host="/h/ws", container=""),
              Mount(host="/h/data", container="/data")]
    assert host_to_container("/h/data/x.gguf", mounts) == "/data/x.gguf"
    assert host_to_container("/h/ws/x.gguf", mounts) is None


def test_first_matching_mount_wins():
    mounts = [Mount(host="/h", container="/root"),
              Mount(host="/h/models", container="/models")]
    assert host_to_container("/h/models/x.gguf", mounts) == "/root/models/x.gguf"
