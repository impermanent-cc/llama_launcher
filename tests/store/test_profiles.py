from pathlib import Path

from llama_launcher.core.spec import LoraRef, Mount, Profile, RouterMember, Runtime
from llama_launcher.store.profiles import (
    delete_profile,
    list_profiles,
    load_config,
    load_profile,
    profile_from_dict,
    profile_to_dict,
    save_config,
    save_profile,
)


def _profile():
    return Profile(
        name="Test One",
        image="img:tag",
        runtime=Runtime(binary="podman", gpu_mode="cdi"),
        mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
        model="/models/m.gguf",
        mmproj="/models/p.gguf",
        loras=[LoraRef(path="/models/l.gguf", scale=0.5)],
        settings={"temp": 0.6, "port": 8080},
        raw_args="--verbose",
    )


def test_dict_round_trip():
    p = _profile()
    assert profile_from_dict(profile_to_dict(p)) == p


def test_save_and_load(tmp_path: Path):
    p = _profile()
    path = save_profile(p, tmp_path)
    assert path.exists()
    assert load_profile(path) == p


def test_stop_timeout_round_trips():
    p = Profile(name="Slow", runtime=Runtime(stop_timeout=45))
    assert profile_from_dict(profile_to_dict(p)).runtime.stop_timeout == 45


def test_stop_timeout_defaults_for_legacy_profile():
    # A profile JSON with no stop_timeout key loads with the 10s default.
    legacy = profile_to_dict(Profile(name="Old"))
    legacy["runtime"].pop("stop_timeout", None)
    assert profile_from_dict(legacy).runtime.stop_timeout == 10


def test_list_and_delete(tmp_path: Path):
    save_profile(_profile(), tmp_path)
    save_profile(Profile(name="Second"), tmp_path)
    names = {p.name for p in list_profiles(tmp_path)}
    assert names == {"Test One", "Second"}
    delete_profile("Test One", tmp_path)
    assert {p.name for p in list_profiles(tmp_path)} == {"Second"}


def test_config_round_trip(tmp_path: Path):
    save_config({"terminal": "konsole", "last_profile": "Test One"}, tmp_path)
    assert load_config(tmp_path)["terminal"] == "konsole"


def test_load_config_missing_returns_empty(tmp_path: Path):
    assert load_config(tmp_path) == {}


def test_router_profile_round_trips(tmp_path):
    p = Profile(
        name="Router",
        mode="router",
        runtime=Runtime(bind_host="0.0.0.0"),
        members=[
            RouterMember(
                profile="Qwen", model_id="qwen", load_on_startup=True, stop_timeout=30
            )
        ],
    )
    back = profile_from_dict(profile_to_dict(p))
    assert back.mode == "router"
    assert back.runtime.bind_host == "0.0.0.0"
    assert back.members == [
        RouterMember(
            profile="Qwen", model_id="qwen", load_on_startup=True, stop_timeout=30
        )
    ]


def test_legacy_profile_json_without_new_fields_still_loads():
    # A profile JSON with no mode/members/bind_host keys loads with defaults.
    back = profile_from_dict({"name": "Old", "model": "/models/a.gguf"})
    assert back.mode == "server"
    assert back.members == []
    assert back.runtime.bind_host == "127.0.0.1"


def _member(profile_name):
    return RouterMember(profile=profile_name)


def test_resolve_member_pairs_pairs_present_and_drops_missing(tmp_path, monkeypatch):
    from llama_launcher.store import profiles as store

    a = Profile(name="a", image="img", runtime=Runtime())
    b = Profile(name="b", image="img", runtime=Runtime())
    monkeypatch.setattr(store, "list_profiles", lambda base: [a, b])
    members = [_member("a"), _member("gone"), _member("b")]
    pairs = store.resolve_member_pairs(members, tmp_path)
    assert [(m.profile, p.name) for m, p in pairs] == [("a", "a"), ("b", "b")]


def test_resolve_member_pairs_empty(tmp_path, monkeypatch):
    from llama_launcher.store import profiles as store

    monkeypatch.setattr(store, "list_profiles", lambda base: [])
    assert store.resolve_member_pairs([_member("x")], tmp_path) == []


def test_detached_round_trips():
    from llama_launcher.core.spec import Profile, Runtime
    from llama_launcher.store.profiles import profile_from_dict, profile_to_dict

    p = Profile(name="Solo", runtime=Runtime(detached=True))
    back = profile_from_dict(profile_to_dict(p))
    assert back.runtime.detached is True


def test_profile_without_detached_key_defaults_false():
    from llama_launcher.store.profiles import profile_from_dict

    # A profile JSON with no runtime.detached key.
    p = profile_from_dict({"name": "Legacy", "runtime": {"binary": "podman"}})
    assert p.runtime.detached is False


def test_launch_mode_and_native_binary_round_trip():
    p = Profile(
        name="Native",
        runtime=Runtime(
            launch_mode="native", native_binary="/opt/llama/build/bin/llama-server"
        ),
    )
    out = profile_from_dict(profile_to_dict(p))
    assert out.runtime.launch_mode == "native"
    assert out.runtime.native_binary == "/opt/llama/build/bin/llama-server"


def test_legacy_profile_defaults_to_container():
    legacy = profile_to_dict(Profile(name="Old"))
    legacy["runtime"].pop("launch_mode", None)
    legacy["runtime"].pop("native_binary", None)
    out = profile_from_dict(legacy)
    assert out.runtime.launch_mode == "container"
    assert out.runtime.native_binary == ""


def test_rpc_workers_round_trip():
    from llama_launcher.core.spec import RpcWorker

    p = Profile(
        name="pool",
        runtime=Runtime(
            launch_mode="rpc",
            rpc_workers=[
                RpcWorker(node="local", device="CUDA0", mem_mb=8000, port=50052),
                RpcWorker(node="box2", device="CPU", mem_mb=32000, port=50052),
            ],
        ),
    )
    back = profile_from_dict(profile_to_dict(p))
    assert back.runtime.launch_mode == "rpc"
    assert back.runtime.rpc_workers == p.runtime.rpc_workers
    assert isinstance(back.runtime.rpc_workers[0], RpcWorker)


def test_container_profile_round_trip_unchanged():
    p = Profile(name="plain", runtime=Runtime())
    assert profile_from_dict(profile_to_dict(p)).runtime.rpc_workers == []


# -- load-time trust clamps ---------------------------------------------------


def test_profile_from_dict_clamps_unknown_binary_to_podman():
    from llama_launcher.store.profiles import profile_from_dict

    p = profile_from_dict({"name": "x", "runtime": {"binary": "/usr/bin/evil"}})
    assert p.runtime.binary == "podman"


def test_profile_from_dict_keeps_podman_and_docker():
    from llama_launcher.store.profiles import profile_from_dict

    assert (
        profile_from_dict({"name": "x", "runtime": {"binary": "docker"}}).runtime.binary
        == "docker"
    )
    assert (
        profile_from_dict({"name": "x", "runtime": {"binary": "podman"}}).runtime.binary
        == "podman"
    )


def test_save_profile_writes_owner_only_permissions(tmp_path):
    """A profile can hold a cleartext api-key; the file (and profiles dir) must
    not be world-readable at the default umask."""
    import stat

    from llama_launcher.core.spec import Profile
    from llama_launcher.store.profiles import save_profile

    p = Profile(name="k", settings={"api-key": "sk-secret"})
    path = save_profile(p, tmp_path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


# -- corruption resilience: one bad file must not brick the GUI ---------------


def test_list_profiles_skips_corrupt_file(tmp_path):
    from llama_launcher.core.spec import Profile
    from llama_launcher.store.profiles import list_profiles, save_profile

    save_profile(Profile(name="good"), tmp_path)
    (tmp_path / "profiles" / "broken.json").write_text("{ this is not json")
    names = [p.name for p in list_profiles(tmp_path)]
    assert "good" in names  # the good one still loads
    assert len(names) == 1  # the broken one is skipped, not fatal


def test_load_config_returns_empty_on_corrupt_file(tmp_path):
    from llama_launcher.store.profiles import load_config

    (tmp_path / "config.json").write_text("{ broken")
    assert load_config(tmp_path) == {}
