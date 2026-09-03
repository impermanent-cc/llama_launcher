import datetime

from llama_launcher.core.build_command import (
    auto_tag,
    config_slug,
    parse_raw_defines,
    render_defines,
)
from llama_launcher.core.build_spec import BuildConfig

D = datetime.date(2026, 8, 28)


def test_slug_normalizes():
    assert config_slug("CUDA Perf build!") == "cuda-perf-build"
    assert config_slug("") == "config"


def test_auto_tag_and_collisions():
    c = BuildConfig(name="cuda perf", engine="llama.cpp")
    assert auto_tag(c, set(), D) == "llama-custom:cuda-perf-20260828"
    taken = {"llama-custom:cuda-perf-20260828"}
    assert auto_tag(c, taken, D) == "llama-custom:cuda-perf-20260828-2"


def test_tag_override_wins():
    c = BuildConfig(name="x", tag_override="me/mine:v1")
    assert auto_tag(c, {"me/mine:v1"}, D) == "me/mine:v1"


def test_render_defines_bool_and_unquoted_tokens():
    # Tokens are UNQUOTED here; the join sites (render_native /
    # render_container) shell-quote each token exactly once.
    c = BuildConfig(options={"cuda": True, "cuda-architectures": "86;120"})
    out = render_defines(c)
    assert "-DGGML_CUDA=ON" in out
    assert "-DCMAKE_CUDA_ARCHITECTURES=86;120" in out


def test_render_native_quotes_special_tokens_once():
    from llama_launcher.core.build_command import render_native

    c = BuildConfig(name="w", source_dir="/s", options={"cuda-architectures": "86;120"})
    assert "'-DCMAKE_CUDA_ARCHITECTURES=86;120'" in render_native(c).configure_cmd


def test_multiword_raw_define_survives_quoted():
    # -DCMAKE_CXX_FLAGS="-O3 -funroll-loops" must stay ONE argument in the
    # copyable command, not split into a broken positional arg.
    from llama_launcher.core.build_command import render_container, render_native

    c = BuildConfig(
        name="perf",
        source_dir="/s",
        raw_defines='-DCMAKE_CXX_FLAGS="-O3 -funroll-loops"',
    )
    assert "'-DCMAKE_CXX_FLAGS=-O3 -funroll-loops'" in render_native(c).configure_cmd
    cf = render_container(
        BuildConfig(
            name="perf",
            builder_image="b",
            runtime_image="r",
            raw_defines='-DCMAKE_CXX_FLAGS="-O3 -funroll-loops"',
        ),
        "t:1",
        "/p/x.containerfile",
    ).containerfile
    assert "'-DCMAKE_CXX_FLAGS=-O3 -funroll-loops'" in cf


def test_build_cmd_quotes_path_with_spaces():
    from llama_launcher.core.build_command import render_container

    c = BuildConfig(name="x", builder_image="b", runtime_image="r")
    cb = render_container(c, "t:1", "/con fig/x.containerfile")
    assert "-f '/con fig/x.containerfile' '/con fig'" in cb.build_cmd


def test_parse_raw_defines_two_token_form():
    # CMake accepts `-D FOO=1`; the value must fold into the define, and a
    # trailing bare -D (still being typed) is dropped, not emitted.
    assert parse_raw_defines("-D FOO=1 -DB=2") == ["-DFOO=1", "-DB=2"]
    assert parse_raw_defines("-D") == []


def test_render_defines_skips_defaults_and_wrong_engine():
    # build-type at its default ("Release") emits nothing; ik-only option on
    # a mainline config emits nothing.
    c = BuildConfig(
        engine="llama.cpp", options={"build-type": "Release", "iqk-fa-all-quants": True}
    )
    assert render_defines(c) == []


def test_raw_defines_win_by_name():
    c = BuildConfig(options={"cuda": True}, raw_defines="-DGGML_CUDA=OFF ignored")
    out = render_defines(c)
    assert out.count("-DGGML_CUDA=OFF") == 1
    assert "-DGGML_CUDA=ON" not in out
    assert "ignored" not in " ".join(out)


def test_parse_raw_defines_filters_non_defines():
    assert parse_raw_defines("-DA=1 rm -rf / -DB=2") == ["-DA=1", "-DB=2"]


def test_render_native_pair():
    from llama_launcher.core.build_command import render_native

    c = BuildConfig(
        name="cuda perf", source_dir="/home/u/src/llama.cpp", options={"cuda": True}
    )
    nb = render_native(c)
    assert nb.configure_cmd == ("cmake -B build-cuda-perf -DGGML_CUDA=ON")
    assert nb.build_cmd == (
        "cmake --build build-cuda-perf -j$(nproc) --target llama-server"
    )
    assert (
        nb.expected_binary == "/home/u/src/llama.cpp/build-cuda-perf/bin/llama-server"
    )


def test_render_native_adds_rpc_target():
    from llama_launcher.core.build_command import render_native

    c = BuildConfig(name="w", source_dir="/s", options={"rpc": True})
    assert "--target llama-server rpc-server" in render_native(c).build_cmd


def test_render_container_structure():
    from llama_launcher.core.build_command import render_container

    c = BuildConfig(
        name="srv",
        engine="ik_llama.cpp",
        target="container",
        options={"cuda": True},
        builder_image="docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04",
        runtime_image="docker.io/nvidia/cuda:12.8.1-runtime-ubuntu24.04",
    )
    cb = render_container(c, "ik-custom:srv-20260828", "/store/srv.containerfile")
    cf = cb.containerfile
    assert cf.startswith("FROM docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build")
    assert "git clone https://github.com/ikawrakow/ik_llama.cpp src" in cf
    assert "git -C src checkout main" in cf  # default branch fallback
    assert "-DGGML_CUDA=ON" in cf
    assert "FROM docker.io/nvidia/cuda:12.8.1-runtime-ubuntu24.04" in cf
    assert 'ENTRYPOINT ["/usr/local/bin/llama-server"]' in cf
    # Build context is the Containerfile's own parent dir, not the caller's
    # CWD: the Containerfile clones its own source, so tarring an unrelated
    # CWD as context is both wrong and wasteful.
    assert (
        cb.build_cmd
        == "podman build -t ik-custom:srv-20260828 -f /store/srv.containerfile /store"
    )


def test_render_container_pinned_ref():
    from llama_launcher.core.build_command import render_container

    c = BuildConfig(name="x", git_ref="b6789", builder_image="b", runtime_image="r")
    cf = render_container(c, "t:1", "/p").containerfile
    assert "git -C src checkout b6789" in cf


def test_default_images_cuda_and_cpu():
    from llama_launcher.core.build_command import default_images

    cuda = BuildConfig(options={"cuda": True})
    assert default_images(cuda) == (
        "docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04",
        "docker.io/nvidia/cuda:12.8.1-runtime-ubuntu24.04",
    )
    cpu = BuildConfig()
    assert default_images(cpu) == (
        "docker.io/library/debian:bookworm",
        "docker.io/library/debian:bookworm-slim",
    )


def test_render_defines_skips_blank_non_bool_values():
    # A cleared string field must emit nothing, never -DGGML_BLAS_VENDOR=''.
    c = BuildConfig(options={"blas-vendor": ""})
    assert render_defines(c) == []


def test_parse_raw_defines_never_raises_on_unbalanced_quote():
    # Fires per keystroke from the Raw defines field; a half-typed quote must
    # degrade to whitespace splitting, not raise ValueError into the Qt slot.
    assert parse_raw_defines('-DFOO="bar') == ['-DFOO="bar']
    assert parse_raw_defines('-DA=1 "') == ["-DA=1"]


def test_rpc_target_detected_for_cmake_spellings():
    from llama_launcher.core.build_command import render_native

    for raw in ("-DGGML_RPC=1", "-DGGML_RPC=on", "-DGGML_RPC:BOOL=TRUE"):
        c = BuildConfig(name="w", source_dir="/s", raw_defines=raw)
        assert "rpc-server" in render_native(c).build_cmd, raw
    c = BuildConfig(name="w", source_dir="/s", raw_defines="-DGGML_RPC=OFF")
    assert "rpc-server" not in render_native(c).build_cmd


def test_rpc_raw_override_of_checkbox_still_builds_target():
    # raw -DGGML_RPC=1 replaces the catalog's =ON rendering via the dedup;
    # the rpc-server target must survive that.
    c = BuildConfig(
        name="w", source_dir="/s", options={"rpc": True}, raw_defines="-DGGML_RPC=1"
    )
    from llama_launcher.core.build_command import render_native

    assert "rpc-server" in render_native(c).build_cmd
