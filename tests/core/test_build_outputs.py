from dataclasses import dataclass

from llama_launcher.core.build_outputs import (
    classify_outputs,
    profiles_using,
    untracked_custom_tags,
)
from llama_launcher.core.build_spec import BuildOutput


@dataclass
class FakeImage:
    tag: str
    size: str = "2 GB"
    created: str = "2026-08-28"


def _out(kind="tag", ident="llama-custom:x-20260828", oid="a1"):
    return BuildOutput(
        id=oid,
        kind=kind,
        identifier=ident,
        config_name="x",
        engine="llama.cpp",
        git_ref="master",
        options={},
        created="2026-08-28",
    )


def test_tag_built_vs_missing():
    outs = [_out(), _out(ident="llama-custom:gone-20260101", oid="b2")]
    images = {"llama-custom:x-20260828": FakeImage("llama-custom:x-20260828")}
    rows = classify_outputs(outs, images, binary_exists=lambda p: False)
    by_id = {r.identifier: r.status for r in rows}
    assert by_id["llama-custom:x-20260828"] == "built"
    assert by_id["llama-custom:gone-20260101"] == "missing"


def test_binary_status_uses_exists_callable():
    outs = [_out(kind="binary", ident="/s/build-x/bin/llama-server")]
    rows = classify_outputs(
        outs, {}, binary_exists=lambda p: p.endswith("llama-server")
    )
    assert rows[0].status == "built"


def test_untracked_custom_tags_only():
    images = {
        "llama-custom:mystery-20260101": FakeImage("llama-custom:mystery-20260101"),
        "ghcr.io/ggml-org/llama.cpp:server": FakeImage("x"),
    }
    assert untracked_custom_tags(images, []) == ["llama-custom:mystery-20260101"]


def test_profiles_using_tag():
    @dataclass
    class P:
        name: str
        image: str

    assert profiles_using("t:1", "tag", [P("a", "t:1"), P("b", "other")]) == ["a"]


def test_profiles_using_tag_localhost_insensitive():
    # use-in-profile on an untracked row writes podman's own localhost/
    # spelling into profile.image; the in-use guard must match it against the
    # registry's unqualified tag (and vice versa) or delete rmi's an image a
    # profile still uses.
    @dataclass
    class P:
        name: str
        image: str

    profs = [P("a", "localhost/llama-custom:x")]
    assert profiles_using("llama-custom:x", "tag", profs) == ["a"]
    assert profiles_using(
        "localhost/llama-custom:x", "tag", [P("b", "llama-custom:x")]
    ) == ["b"]


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
    assert profiles_using(
        "/s/build-x/bin/llama-server", "binary", [p_match, p_no_match]
    ) == ["a"]


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


def test_untracked_custom_tags_registry_port():
    images = {"registry:5000/x-custom:v1": FakeImage("registry:5000/x-custom:v1")}
    assert untracked_custom_tags(images, []) == ["registry:5000/x-custom:v1"]


def test_rootless_localhost_prefix_matches_registry_tag():
    # Rootless podman reports locally built images as localhost/<repo>:<tag>;
    # the registry stores the unqualified tag the user was told to build.
    outs = [_out(ident="llama-custom:x-20260828")]
    images = {
        "localhost/llama-custom:x-20260828": FakeImage(
            "localhost/llama-custom:x-20260828"
        )
    }
    rows = classify_outputs(outs, images, binary_exists=lambda p: False)
    assert rows[0].status == "built"
    # ...and the same image must NOT double-report as untracked.
    assert untracked_custom_tags(images, outs) == []


def test_untracked_reports_podman_spelling():
    # An untracked row must carry podman's own name so rmi works verbatim.
    images = {
        "localhost/ik-custom:mystery-1": FakeImage("localhost/ik-custom:mystery-1")
    }
    assert untracked_custom_tags(images, []) == ["localhost/ik-custom:mystery-1"]
