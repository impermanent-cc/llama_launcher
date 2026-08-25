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


from llama_launcher.core.spec import Profile, RouterMember, Runtime, member_model_id


def test_profile_defaults_to_server_mode():
    p = Profile(name="x")
    assert p.mode == "server"
    assert p.members == []


def test_runtime_defaults_to_loopback_bind():
    assert Runtime().bind_host == "127.0.0.1"


def test_member_model_id_defaults_to_slugified_profile_name():
    assert member_model_id(RouterMember(profile="Qwen3 235B MoE")) == "qwen3-235b-moe"


def test_member_model_id_uses_explicit_override():
    m = RouterMember(profile="Qwen3 235B MoE", model_id="qwen-big")
    assert member_model_id(m) == "qwen-big"


def test_router_member_defaults():
    m = RouterMember(profile="p")
    assert m.load_on_startup is False
    assert m.stop_timeout == 10


def test_runtime_defaults_to_attached():
    from llama_launcher.core.spec import Runtime
    assert Runtime().detached is False


def test_runtime_engine_defaults_to_llama_cpp():
    assert Runtime().engine == "llama.cpp"


def test_runtime_engine_is_settable():
    assert Runtime(engine="ik_llama.cpp").engine == "ik_llama.cpp"


def test_slugify_empty_falls_back_to_placeholder():
    # An all-symbol or empty name must not collapse to "" (a ".json" filename /
    # "llama-" container name); fall back to a stable placeholder.
    assert slugify("!!!") == "unnamed"
    assert slugify("") == "unnamed"
    assert slugify("   ") == "unnamed"
