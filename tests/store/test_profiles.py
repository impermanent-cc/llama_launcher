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
