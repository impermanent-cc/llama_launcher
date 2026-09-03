from llama_launcher.core.spec import Profile, Runtime
from llama_launcher.store import profiles as store


def test_key_mode_defaults_to_global():
    assert Runtime().router_key_mode == "global"


def test_key_mode_round_trips(tmp_path):
    store.save_profile(
        Profile(name="r", image="img", runtime=Runtime(router_key_mode="own")), tmp_path
    )
    loaded = next(p for p in store.list_profiles(tmp_path) if p.name == "r")
    assert loaded.runtime.router_key_mode == "own"


def test_old_profile_without_key_mode_loads_as_global():
    # JSON with no router_key_mode field.
    p = store.profile_from_dict(
        {"name": "r", "image": "img", "runtime": {"binary": "podman"}}
    )
    assert p.runtime.router_key_mode == "global"
