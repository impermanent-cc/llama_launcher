from llama_launcher.core.spec import Profile, Mount, Runtime
from llama_launcher.core.settings_catalog import CATALOG
from llama_launcher.core.command_builder import build_command
from llama_launcher.store.profiles import profile_to_dict, profile_from_dict


def test_new_catalog_settings_present():
    assert CATALOG["metrics"].flag == "--metrics" and CATALOG["metrics"].type == "bool"
    assert CATALOG["no-slots"].flag == "--no-slots"
    assert CATALOG["props"].flag == "--props"
    assert CATALOG["spec-draft-ngl"].type == "int_or_token"
    assert CATALOG["spec-draft-n-max"].default == 3


def test_draft_model_emitted():
    p = Profile(name="p", image="img",
                mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
                model="/models/m.gguf", draft_model="/models/draft.gguf",
                settings={"port": 8080, "spec-draft-n-max": 5})
    argv = build_command(p)
    assert argv[argv.index("--spec-draft-model") + 1] == "/models/draft.gguf"
    assert "--spec-draft-n-max" in argv and "5" in argv


def test_no_draft_no_flag():
    p = Profile(name="p", image="img", model="/models/m.gguf", settings={"port": 8080})
    assert "--spec-draft-model" not in build_command(p)


def test_draft_model_round_trips():
    p = Profile(name="p", draft_model="/models/d.gguf")
    assert profile_from_dict(profile_to_dict(p)).draft_model == "/models/d.gguf"


def test_draft_model_uses_ik_spelling_on_ik_engine():
    """ik_llama.cpp rejects --spec-draft-model; it only accepts -md/--model-draft.

    Probed by execution against ik-llama-cpp:cu12-server, whose parser answers
    "unknown argument: --spec-draft-model". Emitting the mainline spelling killed
    the launch for every ik profile with a draft model set.
    """
    p = Profile(name="p", image="ik-llama-cpp:cu12-server",
                runtime=Runtime(engine="ik_llama.cpp"),
                model="/models/m.gguf", draft_model="/models/draft.gguf",
                settings={"port": 8080})
    argv = build_command(p)
    assert "--spec-draft-model" not in argv
    assert argv[argv.index("--model-draft") + 1] == "/models/draft.gguf"


def test_draft_model_raw_arg_override_folds_across_spellings():
    """A raw -md/--model-draft must override the launcher's own draft flag
    rather than appending a second, conflicting one."""
    p = Profile(name="p", image="img", model="/models/m.gguf",
                draft_model="/models/draft.gguf",
                raw_args="--model-draft /models/other.gguf",
                settings={"port": 8080})
    argv = build_command(p)
    assert argv.count("--spec-draft-model") == 1
    assert "--model-draft" not in argv
    assert argv[argv.index("--spec-draft-model") + 1] == "/models/other.gguf"
