from pathlib import Path

from llama_launcher.core.spec import Profile, Mount, LoraRef, Runtime
from llama_launcher.store.profiles import (
    profile_to_dict, profile_from_dict, save_profile, load_profile,
    list_profiles, delete_profile, load_config, save_config,
)


def _profile():
    return Profile(
        name="Test One", image="img:tag",
        runtime=Runtime(binary="podman", gpu_mode="cdi"),
        mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
        model="/models/m.gguf", mmproj="/models/p.gguf",
        loras=[LoraRef(path="/models/l.gguf", scale=0.5)],
        settings={"temp": 0.6, "port": 8080}, raw_args="--verbose",
    )


def test_dict_round_trip():
    p = _profile()
    assert profile_from_dict(profile_to_dict(p)) == p


def test_save_and_load(tmp_path: Path):
    p = _profile()
    path = save_profile(p, tmp_path)
    assert path.exists()
    assert load_profile(path) == p


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


from llama_launcher.core.spec import Profile, RouterMember, Runtime
from llama_launcher.store.profiles import profile_from_dict, profile_to_dict


def test_router_profile_round_trips(tmp_path):
    p = Profile(
        name="Router",
        mode="router",
        runtime=Runtime(bind_host="0.0.0.0"),
        members=[RouterMember(profile="Qwen", model_id="qwen", load_on_startup=True,
                              stop_timeout=30)],
    )
    back = profile_from_dict(profile_to_dict(p))
    assert back.mode == "router"
    assert back.runtime.bind_host == "0.0.0.0"
    assert back.members == [RouterMember(profile="Qwen", model_id="qwen",
                                         load_on_startup=True, stop_timeout=30)]


def test_legacy_profile_json_without_new_fields_still_loads():
    # Profiles written by earlier versions have no mode/members/bind_host.
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
