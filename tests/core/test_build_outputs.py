from dataclasses import dataclass
from llama_launcher.core.build_spec import BuildOutput
from llama_launcher.core.build_outputs import (
    classify_outputs, untracked_custom_tags, profiles_using, OutputRow,
)


@dataclass
class FakeImage:
    tag: str
    size: str = "2 GB"
    created: str = "2026-08-28"


def _out(kind="tag", ident="llama-custom:x-20260828", oid="a1"):
    return BuildOutput(id=oid, kind=kind, identifier=ident, config_name="x",
                       engine="llama.cpp", git_ref="master", options={},
                       created="2026-08-28")


def test_tag_built_vs_missing():
    outs = [_out(), _out(ident="llama-custom:gone-20260101", oid="b2")]
    images = {"llama-custom:x-20260828": FakeImage("llama-custom:x-20260828")}
    rows = classify_outputs(outs, images, binary_exists=lambda p: False)
    by_id = {r.identifier: r.status for r in rows}
    assert by_id["llama-custom:x-20260828"] == "built"
    assert by_id["llama-custom:gone-20260101"] == "missing"


def test_binary_status_uses_exists_callable():
    outs = [_out(kind="binary", ident="/s/build-x/bin/llama-server")]
    rows = classify_outputs(outs, {}, binary_exists=lambda p: p.endswith("llama-server"))
    assert rows[0].status == "built"


def test_untracked_custom_tags_only():
    images = {"llama-custom:mystery-20260101": FakeImage("llama-custom:mystery-20260101"),
              "ghcr.io/ggml-org/llama.cpp:server": FakeImage("x")}
    assert untracked_custom_tags(images, []) == ["llama-custom:mystery-20260101"]


def test_profiles_using_tag():
    @dataclass
    class P:
        name: str
        image: str
    assert profiles_using("t:1", "tag", [P("a", "t:1"), P("b", "other")]) == ["a"]


def test_profiles_using_binary_direct_match():
    @dataclass
    class R:
        native_binary: str
    @dataclass
    class P:
        name: str
        runtime: R
    p_match = P("a", R("/s/build-x/bin/llama-server"))
    p_no_match = P("b", R("/s/build-y/bin/llama-server"))
    assert profiles_using("/s/build-x/bin/llama-server", "binary", [p_match, p_no_match]) == ["a"]


def test_profiles_using_binary_build_dir_containment():
    @dataclass
    class R:
        native_binary: str
    @dataclass
    class P:
        name: str
        runtime: R
    # Both binaries are in /s/build-x/, so they share the same build dir
    p_match = P("a", R("/s/build-x/bin/llama-server"))
    # Identifier with /s/build-x/something should match
    result = profiles_using("/s/build-x/other/path", "binary", [p_match])
    assert result == ["a"]
