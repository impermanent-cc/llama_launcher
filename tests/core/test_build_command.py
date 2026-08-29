import datetime
from llama_launcher.core.build_spec import BuildConfig
from llama_launcher.core.build_command import (
    config_slug, auto_tag, render_defines, parse_raw_defines,
)

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


def test_render_defines_bool_and_quoting():
    c = BuildConfig(options={"cuda": True, "cuda-architectures": "86;120"})
    out = render_defines(c)
    assert "-DGGML_CUDA=ON" in out
    assert "-DCMAKE_CUDA_ARCHITECTURES='86;120'" in out


def test_render_defines_skips_defaults_and_wrong_engine():
    # build-type at its default ("Release") emits nothing; ik-only option on
    # a mainline config emits nothing.
    c = BuildConfig(engine="llama.cpp",
                    options={"build-type": "Release", "iqk-fa-all-quants": True})
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
    c = BuildConfig(name="cuda perf", source_dir="/home/u/src/llama.cpp",
                    options={"cuda": True})
    nb = render_native(c)
    assert nb.configure_cmd == (
        "cmake -B build-cuda-perf -DGGML_CUDA=ON")
    assert nb.build_cmd == (
        "cmake --build build-cuda-perf -j$(nproc) --target llama-server")
    assert nb.expected_binary == \
        "/home/u/src/llama.cpp/build-cuda-perf/bin/llama-server"


def test_render_native_adds_rpc_target():
    from llama_launcher.core.build_command import render_native
    c = BuildConfig(name="w", source_dir="/s", options={"rpc": True})
    assert "--target llama-server rpc-server" in render_native(c).build_cmd


def test_render_container_structure():
    from llama_launcher.core.build_command import render_container
    c = BuildConfig(name="srv", engine="ik_llama.cpp", target="container",
                    options={"cuda": True},
                    builder_image="docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04",
                    runtime_image="docker.io/nvidia/cuda:12.8.1-runtime-ubuntu24.04")
    cb = render_container(c, "ik-custom:srv-20260828", "/store/srv.containerfile")
    cf = cb.containerfile
    assert cf.startswith("FROM docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build")
    assert "git clone https://github.com/ikawrakow/ik_llama.cpp src" in cf
    assert "git -C src checkout main" in cf          # default branch fallback
    assert "-DGGML_CUDA=ON" in cf
    assert "FROM docker.io/nvidia/cuda:12.8.1-runtime-ubuntu24.04" in cf
    assert 'ENTRYPOINT ["/usr/local/bin/llama-server"]' in cf
    assert cb.build_cmd == \
        "podman build -t ik-custom:srv-20260828 -f /store/srv.containerfile ."


def test_render_container_pinned_ref():
    from llama_launcher.core.build_command import render_container
    c = BuildConfig(name="x", git_ref="b6789",
                    builder_image="b", runtime_image="r")
    cf = render_container(c, "t:1", "/p").containerfile
    assert "git -C src checkout b6789" in cf


def test_default_images_cuda_and_cpu():
    from llama_launcher.core.build_command import default_images
    cuda = BuildConfig(options={"cuda": True})
    assert default_images(cuda) == (
        "docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04",
        "docker.io/nvidia/cuda:12.8.1-runtime-ubuntu24.04")
    cpu = BuildConfig()
    assert default_images(cpu) == (
        "docker.io/library/debian:bookworm",
        "docker.io/library/debian:bookworm-slim")
