from llama_launcher.core.spec import Profile, Runtime
from llama_launcher.store.profiles import profile_from_dict, profile_to_dict


def test_runtime_node_defaults_to_local():
    assert Runtime().node == "local"


def test_old_profile_json_without_node_loads_as_local():
    d = profile_to_dict(Profile(name="p"))
    d["runtime"].pop("node", None)  # a profile file with no node field
    assert profile_from_dict(d).runtime.node == "local"


def test_node_round_trips():
    p = Profile(name="p", runtime=Runtime(node="box-b"))
    assert profile_from_dict(profile_to_dict(p)).runtime.node == "box-b"
