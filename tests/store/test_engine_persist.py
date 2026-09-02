from llama_launcher.core.spec import Profile, Runtime
from llama_launcher.store.profiles import profile_to_dict, profile_from_dict


def test_old_profile_without_engine_loads_as_llama_cpp():
    # Profile JSON with no engine field.
    d = {"name": "old", "runtime": {"binary": "podman"}}
    assert profile_from_dict(d).runtime.engine == "llama.cpp"


def test_engine_survives_roundtrip():
    p = Profile(name="p", runtime=Runtime(engine="ik_llama.cpp"))
    assert profile_from_dict(profile_to_dict(p)).runtime.engine == "ik_llama.cpp"
