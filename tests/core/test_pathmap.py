from llama_launcher.core.spec import Mount
from llama_launcher.core.pathmap import host_to_container, container_to_host


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


def test_container_maps_back_to_host():
    mounts = [Mount(host="/h/models", container="/models")]
    assert container_to_host("/models/q4.gguf", mounts) == "/h/models/q4.gguf"


def test_container_exact_match():
    mounts = [Mount(host="/h/models", container="/models")]
    assert container_to_host("/models", mounts) == "/h/models"


def test_container_not_under_mount_is_none():
    mounts = [Mount(host="/h/models", container="/models")]
    assert container_to_host("/elsewhere/x.gguf", mounts) is None


def test_container_round_trips_with_host_to_container():
    mounts = [Mount(host="/mnt/storage/AI/Models", container="/Models")]
    host = "/mnt/storage/AI/Models/Qwen/m.gguf"
    assert container_to_host(host_to_container(host, mounts), mounts) == host


def test_container_skips_empty_mounts():
    mounts = [Mount(host="", container="/models"),
              Mount(host="/h/data", container="/data")]
    assert container_to_host("/data/x.gguf", mounts) == "/h/data/x.gguf"
    assert container_to_host("/models/x.gguf", mounts) is None
