from llama_launcher.core.command_builder import (
    CONTAINER_KEY_PATH,
    CONTAINER_PRESET_PATH,
    build_command,
)
from llama_launcher.core.spec import Mount, Profile, RouterMember, Runtime


def _router(**kw):
    base = dict(
        name="Host",
        mode="router",
        image="ghcr.io/ggml-org/llama.cpp:server-cuda12-b10290",
        mounts=[Mount(host="/mnt/storage/AI/Models", container="/models")],
        members=[RouterMember(profile="Qwen")],
        settings={"port": 8080},
    )
    base.update(kw)
    return Profile(**base)


def test_router_runs_detached_without_rm():
    argv = build_command(_router(), router_host_dir="/cfg/router/host")
    assert "-d" in argv
    # A headless host that dies overnight must leave an inspectable container.
    assert "--rm" not in argv


def test_router_passes_no_model_flag():
    argv = build_command(_router(), router_host_dir="/cfg/router/host")
    assert "-m" not in argv


def test_router_mounts_config_dir_read_only():
    argv = build_command(_router(), router_host_dir="/cfg/router/host")
    assert "/cfg/router/host:/router:ro" in argv


def test_router_passes_preset_and_key_file():
    argv = build_command(_router(), router_host_dir="/cfg/router/host")
    assert argv[argv.index("--models-preset") + 1] == CONTAINER_PRESET_PATH
    assert argv[argv.index("--api-key-file") + 1] == CONTAINER_KEY_PATH


def test_router_publishes_on_bind_host():
    p = _router(runtime=Runtime(bind_host="0.0.0.0"))
    argv = build_command(p, router_host_dir="/cfg/r")
    assert "0.0.0.0:8080:8080" in argv


def test_containers_are_labelled_for_reattach():
    argv = build_command(_router(), router_host_dir="/cfg/r")
    assert "llama-launcher.profile=Host" in argv
    assert "llama-launcher.mode=router" in argv


def test_router_emits_host_level_settings_only():
    p = _router(settings={"port": 8080, "models-max": 2, "ctx-size": 4096})
    argv = build_command(p, router_host_dir="/cfg/r")
    assert argv[argv.index("--models-max") + 1] == "2"
    # ctx-size is model-level: it would override every member's own value.
    assert "--ctx-size" not in argv


def test_server_mode_is_unchanged():
    p = Profile(
        name="Solo",
        image="img",
        model="/models/a.gguf",
        mounts=[Mount(host="/h", container="/models")],
        settings={"port": 8080},
    )
    argv = build_command(p)
    assert "--rm" in argv
    assert "-d" not in argv
    assert argv[argv.index("-m") + 1] == "/models/a.gguf"
    assert "127.0.0.1:8080:8080" in argv
    assert "llama-launcher.mode=server" in argv


def test_models_max_is_emitted_when_set_away_from_upstream_default():
    argv = build_command(
        _router(settings={"port": 8080, "models-max": 1}), router_host_dir="/cfg/r"
    )
    assert argv[argv.index("--models-max") + 1] == "1"


def test_disabling_autoload_actually_emits_a_flag():
    argv = build_command(
        _router(settings={"port": 8080, "models-autoload": True}),
        router_host_dir="/cfg/r",
    )
    assert "--no-models-autoload" in argv


def test_api_key_setting_never_reaches_router_argv():
    # The whole point of --api-key-file: argv is visible in podman inspect, ps,
    # the exported .sh and the diagnostic report.
    argv = build_command(
        _router(settings={"port": 8080, "api-key": "sk-LEAKED"}),
        router_host_dir="/cfg/r",
    )
    assert not any("sk-LEAKED" in a for a in argv)
    assert "--api-key" not in argv


def test_server_args_never_emit_router_only_flags():
    # Defence in depth: the UI filters by mode, but a Profile loaded from JSON
    # can still carry the key.
    p = Profile(
        name="Solo",
        image="img",
        model="/models/a.gguf",
        mounts=[Mount(host="/h", container="/models")],
        settings={"port": 8080, "models-max": 4, "models-autoload": True},
    )
    argv = build_command(p)
    assert "--models-max" not in argv
    assert "--no-models-autoload" not in argv


def _router_raw(raw="", **settings):
    return Profile(
        name="r",
        image="img",
        runtime=Runtime(bind_host="127.0.0.1"),
        mode="router",
        settings={"port": 8080, **settings},
        raw_args=raw,
    )


def test_router_raw_overrides_owned_setting():
    argv = build_command(_router_raw(raw="--models-max 3", **{"models-max": 1}))
    assert argv.count("--models-max") == 1
    assert argv[argv.index("--models-max") + 1] == "3"  # raw wins in place


def test_router_raw_models_preset_cannot_override():
    argv = build_command(_router_raw(raw="--models-preset /evil.ini"))
    assert argv.count("--models-preset") == 1
    assert "/evil.ini" not in argv  # launcher keeps its container path


def test_router_noncolliding_raw_appended():
    argv = build_command(_router_raw(raw="--numa distribute"))
    assert argv[-2:] == ["--numa", "distribute"]
