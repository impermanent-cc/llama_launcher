from llama_launcher.core.build_spec import (
    BuildConfig, BuildOutput,
    build_config_to_dict, build_config_from_dict,
    build_output_to_dict, build_output_from_dict,
)


def test_build_config_roundtrip():
    c = BuildConfig(name="cuda perf", engine="ik_llama.cpp", target="container",
                    git_ref="v1.2", options={"ggml-cuda": True}, raw_defines="-DX=1")
    assert build_config_from_dict(build_config_to_dict(c)) == c


def test_build_config_from_dict_ignores_unknown_and_fills_missing():
    c = build_config_from_dict({"name": "n", "bogus_key": 1})
    assert c.name == "n"
    assert c.engine == "llama.cpp"
    assert c.options == {}


def test_build_output_roundtrip():
    o = BuildOutput(id="abc123", kind="tag", identifier="llama-custom:x-20260828",
                    config_name="x", engine="llama.cpp", git_ref="master",
                    options={"ggml-cuda": True}, created="2026-08-28")
    assert build_output_from_dict(build_output_to_dict(o)) == o
