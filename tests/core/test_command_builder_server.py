from llama_launcher.core.spec import Profile, Mount, Runtime, LoraRef
from llama_launcher.core.command_builder import build_command


def _golden_profile():
    return Profile(
        name="Qwen3-235B coding",
        image="ghcr.io/ggml-org/llama.cpp:server-cuda12-b9628",
        runtime=Runtime(binary="podman", gpu_mode="cdi"),
        mounts=[Mount(host="/mnt/storage/AI/Models", container="/models",
                      role="model", mode="ro")],
        model="/models/modelpath/",
        settings={
            "n-gpu-layers": 99, "n-cpu-moe": 20, "flash-attn": "on",
            "ctx-size": 65536, "cache-type-k": "q8_0", "cache-type-v": "q8_0",
            "threads": 18, "temp": 0.6, "top-p": 0.95, "top-k": 20, "min-p": 0.0,
            "repeat-penalty": 1.0, "dry-multiplier": 0.8, "dry-base": 1.75,
            "dry-allowed-length": 2, "ubatch-size": 512, "cache-reuse": 256,
            "parallel": 1, "no-mmap": True, "jinja": True, "tools": "all",
            "port": 8080,
        },
    )


def _pairs(argv):
    """Return set of (flag, value) for `--flag value` and bare flags."""
    pairs = set()
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            pairs.add((tok, argv[i + 1]))
            i += 2
        else:
            pairs.add((tok,))
            i += 1
    return pairs


def test_model_after_image():
    argv = build_command(_golden_profile())
    assert "-m" in argv
    assert argv[argv.index("-m") + 1] == "/models/modelpath/"


def test_host_injected_and_port_present():
    argv = build_command(_golden_profile())
    assert argv[argv.index("--host") + 1] == "0.0.0.0"
    assert argv[argv.index("--port") + 1] == "8080"


def test_changed_settings_rendered():
    argv = build_command(_golden_profile())
    text = " ".join(argv)
    for frag in ["--ctx-size 65536", "--flash-attn on", "--cache-type-k q8_0",
                 "--n-cpu-moe 20", "--temp 0.6", "--top-k 20", "--cache-reuse 256",
                 "--tools all", "--n-gpu-layers 99"]:
        assert frag in text, frag
    # bool flags are bare
    assert "--no-mmap" in argv
    assert "--jinja" in argv


def test_settings_emitted_in_catalog_order():
    argv = build_command(_golden_profile())
    # ctx-size (Model&Context) precedes n-gpu-layers (GPU) precedes temp (Sampling)
    assert argv.index("--ctx-size") < argv.index("--n-gpu-layers") < argv.index("--temp")


def test_deterministic():
    p = _golden_profile()
    assert build_command(p) == build_command(p)


def test_mmproj_and_loras():
    p = _golden_profile()
    p.mmproj = "/models/mmproj.gguf"
    p.loras = [LoraRef(path="/models/a.gguf", scale=1.0),
               LoraRef(path="/models/b.gguf", scale=0.5)]
    argv = build_command(p)
    assert argv[argv.index("--mmproj") + 1] == "/models/mmproj.gguf"
    assert "/models/a.gguf" in argv               # scale 1.0 -> plain --lora
    assert argv[argv.index("--lora") + 1] == "/models/a.gguf"
    assert "--lora-scaled" in argv
    assert "/models/b.gguf:0.5" in argv


def test_raw_args_appended_last():
    p = _golden_profile()
    p.raw_args = "--verbose --override-kv foo=int:1"
    argv = build_command(p)
    assert argv[-3:] == ["--verbose", "--override-kv", "foo=int:1"]


def test_int_or_token_value_all():
    p = _golden_profile()
    p.settings["n-gpu-layers"] = "all"
    argv = build_command(p)
    assert argv[argv.index("--n-gpu-layers") + 1] == "all"


def test_spec_type_mtp_emitted():
    p = _golden_profile()
    p.settings["spec-type"] = "draft-mtp"
    assert "--spec-type draft-mtp" in " ".join(build_command(p))


def test_spec_type_absent_when_unset():
    argv = build_command(_golden_profile())   # no spec-type key
    assert "--spec-type" not in argv


def test_draft_cache_types_render():
    p = _golden_profile()
    p.settings["cache-type-k-draft"] = "q8_0"
    p.settings["spec-draft-n-min"] = 1
    text = " ".join(build_command(p))
    assert "--cache-type-k-draft q8_0" in text
    assert "--spec-draft-n-min 1" in text
