from llama_launcher.core.spec import Mount, LoraRef, Runtime, Profile, slugify


def test_profile_defaults():
    p = Profile(name="My Model")
    assert p.runtime.binary == "podman"
    assert p.runtime.gpu_mode == "cdi"
    assert p.mounts == []
    assert p.loras == []
    assert p.settings == {}
    assert p.mmproj is None


def test_mount_defaults():
    m = Mount(host="/h", container="/c")
    assert m.mode == "ro"
    assert m.role == "custom"
    assert m.selinux is None
    assert m.workdir is False


def test_independent_default_collections():
    a = Profile(name="a")
    b = Profile(name="b")
    a.mounts.append(Mount(host="/h", container="/c"))
    assert b.mounts == []  # no shared mutable default


def test_slugify():
    assert slugify("Qwen3-235B coding!") == "qwen3-235b-coding"
    assert slugify("  multiple   spaces ") == "multiple-spaces"
