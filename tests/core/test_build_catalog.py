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
    assert {s.engine for s in BUILD_CATALOG.values()} <= {
        "any",
        "llama.cpp",
        "ik_llama.cpp",
    }


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
    assert (
        set(REPO_URL)
        == set(DEFAULT_BRANCH)
        == set(ENGINE_SHORT)
        == {"llama.cpp", "ik_llama.cpp"}
    )
    assert DEFAULT_BRANCH["llama.cpp"] == "master"
    assert DEFAULT_BRANCH["ik_llama.cpp"] == "main"
