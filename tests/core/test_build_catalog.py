from llama_launcher.core.build_catalog import (
    BUILD_CATALOG,
    DEFAULT_BRANCH,
    ENGINE_SHORT,
    REPO_URL,
)
from llama_launcher.core.settings_catalog import for_engine


def test_keys_match_setting_key_and_no_aliases():
    for key, s in BUILD_CATALOG.items():
        assert s.key == key
        assert s.aliases == ()  # cmake defines have no alias spellings
        assert not s.flag.startswith("-")  # bare variable name, -D added at render


def test_engine_values_are_valid():
    from llama_launcher.core.settings_catalog import engines_of
    from llama_launcher.core.spec import ENGINES

    for s in BUILD_CATALOG.values():
        assert set(engines_of(s)) <= set(ENGINES), s.key


def test_for_engine_filters_both_directions():
    ml = for_engine(BUILD_CATALOG, "llama.cpp")
    ik = for_engine(BUILD_CATALOG, "ik_llama.cpp")
    assert "iqk-flash-attention" in ik and "iqk-flash-attention" not in ml
    # GGML_CPU_REPACK is mainline-only (ik's equivalent is run-time -rtr)
    assert "cpu-repack" in ml and "cpu-repack" not in ik


def test_enum_defaults_within_enum():
    for key, s in BUILD_CATALOG.items():
        if s.type == "enum":
            assert s.default in s.enum, key


def test_core_entries_present_with_expected_flags():
    assert BUILD_CATALOG["cuda"].flag == "GGML_CUDA"
    assert BUILD_CATALOG["cuda-architectures"].flag == "CMAKE_CUDA_ARCHITECTURES"
    assert BUILD_CATALOG["build-type"].flag == "CMAKE_BUILD_TYPE"
    assert BUILD_CATALOG["build-type"].default == "Release"
    assert BUILD_CATALOG["rpc"].flag == "GGML_RPC"
    assert BUILD_CATALOG["native-opt"].flag == "GGML_NATIVE"


def test_sched_max_copies_split_by_engine():
    # Each engine gets its own gated Setting for the same CMake flag: one
    # "any"-gated entry can carry only one default, and ik's (1) differs
    # from mainline's (4).
    ml = for_engine(BUILD_CATALOG, "llama.cpp")
    ik = for_engine(BUILD_CATALOG, "ik_llama.cpp")

    assert ml["sched-max-copies"].flag == "GGML_SCHED_MAX_COPIES"
    assert ml["sched-max-copies"].default == 4
    assert "sched-max-copies-ik" not in ml

    assert ik["sched-max-copies-ik"].flag == "GGML_SCHED_MAX_COPIES"
    assert ik["sched-max-copies-ik"].default == 1
    assert "sched-max-copies" not in ik


def test_repo_constants():
    from llama_launcher.core.spec import ENGINES

    assert set(REPO_URL) == set(DEFAULT_BRANCH) == set(ENGINE_SHORT) == set(ENGINES)
    assert DEFAULT_BRANCH["llama.cpp"] == "master"
    assert DEFAULT_BRANCH["ik_llama.cpp"] == "main"
    assert DEFAULT_BRANCH["prism_llama.cpp"] == "prism"
    assert REPO_URL["prism_llama.cpp"] == "https://github.com/PrismML-Eng/llama.cpp"
    assert ENGINE_SHORT["prism_llama.cpp"] == "prism"


def test_peer_max_batch_size_is_ik_only():
    # Mainline ggml does not define this cache variable; ik_llama.cpp does.
    ml = for_engine(BUILD_CATALOG, "llama.cpp")
    ik = for_engine(BUILD_CATALOG, "ik_llama.cpp")
    assert "cuda-peer-max-batch-size" in ik
    assert "cuda-peer-max-batch-size" not in ml


def test_every_engine_gated_option_carries_a_tooltip():
    gated = [o for o in BUILD_CATALOG.values() if o.engine != "any"]
    assert gated and all(o.tooltip.strip() for o in gated), [
        o.key for o in gated if not o.tooltip.strip()
    ]


def test_prism_shares_mainline_cmake_options_plus_its_hopper_path():
    """PrismML's CMake files are mainline's plus one option, so every
    mainline-only option reaches it, no ik-only option does, and its own
    option reaches it alone."""
    ml = for_engine(BUILD_CATALOG, "llama.cpp")
    ik = for_engine(BUILD_CATALOG, "ik_llama.cpp")
    prism = for_engine(BUILD_CATALOG, "prism_llama.cpp")
    assert set(prism) == (set(ml) | {"cuda-hopper-q1"})
    assert "iqk-flash-attention" not in prism
    assert "cuda-hopper-q1" not in ml and "cuda-hopper-q1" not in ik
    hopper = BUILD_CATALOG["cuda-hopper-q1"]
    assert hopper.flag == "GGML_CUDA_HOPPER_Q1"
    assert hopper.type == "bool" and hopper.default is False
    assert hopper.engine == "prism_llama.cpp"
    assert prism["sched-max-copies"].default == 4
