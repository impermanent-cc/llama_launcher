from dataclasses import replace

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


def test_mmproj_device_string_flag():
    p = _golden_profile()
    argv = build_command(replace(p, settings={**p.settings, "mmproj-device": "CUDA0"}))
    assert argv[argv.index("--mmproj-device") + 1] == "CUDA0"
    # empty string is omitted, like other string flags
    argv2 = build_command(replace(p, settings={**p.settings, "mmproj-device": ""}))
    assert "--mmproj-device" not in argv2


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


def test_mmproj_offload_and_override_tensor_render():
    p = _golden_profile()
    p.settings["no-mmproj-offload"] = True
    p.settings["override-tensor"] = "exps=CPU"
    p.settings["split-mode"] = "tensor"
    argv = build_command(p)
    text = " ".join(argv)
    assert "--no-mmproj-offload" in argv          # bare bool flag
    assert "--override-tensor exps=CPU" in text
    assert "--split-mode tensor" in text


def test_swa_and_checkpoint_render():
    p = _golden_profile()
    p.settings["swa-full"] = True
    p.settings["ctx-checkpoints"] = 8
    p.settings["checkpoint-min-step"] = 512
    argv = build_command(p)
    text = " ".join(argv)
    assert "--swa-full" in argv
    assert "--ctx-checkpoints 8" in text
    assert "--checkpoint-min-step 512" in text


def test_numa_value_and_server_round_out_render():
    p = _golden_profile()
    p.settings["numa"] = "distribute"
    p.settings["threads-http"] = 8
    p.settings["no-webui"] = True
    p.settings["reasoning-format"] = "deepseek"
    p.settings["dry-sequence-breaker"] = "none"
    text = " ".join(build_command(p))
    assert "--numa distribute" in text
    assert "--threads-http 8" in text
    assert "--no-webui" in build_command(p)
    assert "--reasoning-format deepseek" in text
    assert "--dry-sequence-breaker none" in text


def test_chat_template_kwargs_json_value_reaches_argv_intact():
    p = _golden_profile()
    p.settings["chat-template-kwargs"] = '{"enable_thinking":false}'
    argv = build_command(p)
    i = argv.index("--chat-template-kwargs")
    assert argv[i + 1] == '{"enable_thinking":false}'


def test_reasoning_budget_message_reaches_argv():
    p = _golden_profile()
    p.settings["reasoning-budget-message"] = "Thinking budget reached."
    argv = build_command(p)
    i = argv.index("--reasoning-budget-message")
    assert argv[i + 1] == "Thinking budget reached."


def test_spec_draft_backend_sampling_emits_no_flag_only_when_checked():
    p = _golden_profile()
    assert "--no-spec-draft-backend-sampling" not in build_command(p)
    p.settings["spec-draft-backend-sampling"] = True
    assert "--no-spec-draft-backend-sampling" in build_command(p)


def _embed_profile(**settings):
    return Profile(
        name="embed", image="ghcr.io/ggml-org/llama.cpp:server-cuda12-b9628",
        runtime=Runtime(binary="podman", gpu_mode="cdi"),
        mounts=[Mount(host="/mnt/storage/AI/Models", container="/models",
                      role="model", mode="ro")],
        model="/models/nomic-embed.gguf",
        settings={"port": 8080, **settings},
    )


def test_embedding_flags_render():
    argv = build_command(_embed_profile(embeddings=True, pooling="mean"))
    assert "--embeddings" in argv
    i = argv.index("--pooling")
    assert argv[i + 1] == "mean"
    assert "--reranking" not in argv


def test_reranking_flags_render():
    argv = build_command(_embed_profile(reranking=True, pooling="rank", embeddings=True))
    assert "--reranking" in argv
    assert "--embeddings" in argv
    i = argv.index("--pooling")
    assert argv[i + 1] == "rank"


def test_pooling_absent_emits_no_flag():
    argv = build_command(_embed_profile(embeddings=True))
    assert "--pooling" not in argv


def _srv(raw="", **settings):
    return Profile(
        name="s", image="img", runtime=Runtime(bind_host="127.0.0.1"), mode="server",
        mounts=[Mount(host="/h", container="/models", role="model")],
        model="/models/m.gguf", raw_args=raw,
        settings={"port": 8080, **settings},
    )


def test_server_raw_ngl_overrides_setting_no_duplicate():
    argv = build_command(_srv(raw="-ngl 50", **{"n-gpu-layers": "99"}))
    assert argv.count("--n-gpu-layers") == 1
    i = argv.index("--n-gpu-layers")
    assert argv[i + 1] == "50"
    assert "-ngl" not in argv           # raw short form folded onto the owned long form


def test_server_raw_port_cannot_override():
    argv = build_command(_srv(raw="--port 9000"))
    assert argv.count("--port") == 1
    assert "9000" not in argv           # launcher keeps 8080


def test_server_noncolliding_raw_still_appended():
    argv = build_command(_srv(raw="--numa distribute"))
    assert argv[-2:] == ["--numa", "distribute"]


def test_server_multiple_loras_preserved_with_raw_lora():
    p = _srv(raw="--lora /r.gguf")
    from llama_launcher.core.spec import LoraRef
    p.loras.append(LoraRef(path="/models/base.gguf"))
    argv = build_command(p)
    assert argv.count("--lora") == 2    # owned lora + raw lora both present


def test_server_empty_string_setting_emits_nothing():
    # Clearing a string field whose default is non-empty (e.g. cors-origins,
    # default "*") stores "" -- distinct from the default -- but an empty value
    # is meaningless as a flag argument and must not reach argv.
    argv = build_command(_srv(**{"cors-origins": ""}))
    assert "--cors-origins" not in argv


def test_server_whitespace_only_string_setting_emits_nothing():
    argv = build_command(_srv(**{"cors-origins": "   "}))
    assert "--cors-origins" not in argv


def test_server_nonempty_string_setting_still_emitted():
    argv = build_command(_srv(**{"cors-origins": "http://localhost:3000"}))
    assert argv[argv.index("--cors-origins") + 1] == "http://localhost:3000"


def test_server_empty_multiselect_from_json_emits_nothing():
    # The CLI/headless path bypasses the UI's default-gating, so a profile JSON
    # carrying an empty multiselect (tools="") reaches build_command directly.
    argv = build_command(_srv(**{"tools": ""}))
    assert "--tools" not in argv


def test_server_empty_enum_from_json_emits_nothing():
    argv = build_command(_srv(**{"flash-attn": ""}))
    assert "--flash-attn" not in argv


def test_server_numeric_zero_setting_preserved():
    # Guard must not swallow legitimate 0 / -1 values -- only blank strings.
    argv = build_command(_srv(**{"min-p": 0.0, "sleep-idle-seconds": -1}))
    assert argv[argv.index("--min-p") + 1] == "0.0"
    assert argv[argv.index("--sleep-idle-seconds") + 1] == "-1"


def test_load_mode_emitted():
    argv = build_command(_srv(**{"load-mode": "mmap+mlock"}))
    assert argv[argv.index("--load-mode") + 1] == "mmap+mlock"


def test_load_mode_suppresses_legacy_mmap_mlock():
    # When load-mode is set, the deprecated --no-mmap/--mlock must not also emit
    # (llama.cpp warns and honours only the last of the two).
    argv = build_command(_srv(**{"load-mode": "none", "no-mmap": True, "mlock": True}))
    assert "--load-mode" in argv
    assert "--no-mmap" not in argv
    assert "--mlock" not in argv


def test_legacy_mmap_mlock_still_emit_without_load_mode():
    argv = build_command(_srv(**{"no-mmap": True, "mlock": True}))
    assert "--no-mmap" in argv
    assert "--mlock" in argv
    assert "--load-mode" not in argv


def test_load_mode_mmap_emits_and_suppresses_legacy():
    # The default is "auto", so an explicit mmap choice is non-default:
    # it must emit and still win over the legacy pair.
    argv = build_command(_srv(**{"load-mode": "mmap", "no-mmap": True}))
    assert argv[argv.index("--load-mode") + 1] == "mmap"
    assert "--no-mmap" not in argv


def test_load_mode_at_default_does_not_suppress_legacy():
    # A JSON leftover load-mode equal to its default emits nothing, so it must
    # not eat the legacy flags either (nothing would carry the intent at all).
    argv = build_command(_srv(**{"load-mode": "auto", "no-mmap": True}))
    assert "--load-mode" not in argv
    assert "--no-mmap" in argv


def _ik_profile(engine):
    return Profile(
        name="p", image="ghcr.io/ikawrakow/ik-llama-cpp:cu12-server",
        runtime=Runtime(engine=engine),
        mounts=[Mount(host="/m", container="/models", role="model")],
        model="/models/x.gguf",
        settings={"run-time-repack": True, "port": 8080},
    )


def test_ik_flag_emitted_when_engine_is_ik():
    argv = build_command(_ik_profile("ik_llama.cpp"))
    assert "--run-time-repack" in argv


def test_ik_flag_never_emitted_on_llama_cpp_engine():
    # Same settings dict, mainline engine (JSON-leftover scenario) -> dropped.
    argv = build_command(_ik_profile("llama.cpp"))
    assert "--run-time-repack" not in argv


def _ik_srv(**settings):
    return Profile(
        name="p", image="ik-img", runtime=Runtime(engine="ik_llama.cpp"),
        mounts=[Mount(host="/m", container="/models", role="model")],
        model="/models/x.gguf", settings={"port": 8080, **settings},
    )


def test_spec_type_translated_for_ik_engine():
    # ik's parser wants 'dflash', not mainline's 'draft-dflash'.
    argv = build_command(_ik_srv(**{"spec-type": "draft-dflash"}))
    assert argv[argv.index("--spec-type") + 1] == "dflash"


def test_spec_type_untranslated_on_mainline():
    argv = build_command(_srv(**{"spec-type": "draft-dflash"}))
    assert argv[argv.index("--spec-type") + 1] == "draft-dflash"


def test_spec_type_ngram_passes_through_on_ik():
    # ik normalizes '-' to '_' itself; ngram names need no translation.
    argv = build_command(_ik_srv(**{"spec-type": "ngram-map-k"}))
    assert argv[argv.index("--spec-type") + 1] == "ngram-map-k"


def test_ik_suffix_spec_type_dropped_on_mainline():
    # 'suffix' exists only in ik's parser; a JSON leftover must not reach a
    # mainline launch (same guard as the ik-only KV-cache quant values).
    argv = build_command(_srv(**{"spec-type": "suffix"}))
    assert "--spec-type" not in argv


def test_amb_zero_emits_explicit_no_cap():
    # ik's -amb default is 256; 0 (= no cap) is non-default and must emit.
    argv = build_command(_ik_srv(**{"attention-max-batch": 0}))
    assert argv[argv.index("--attention-max-batch") + 1] == "0"


def test_ik_kv_enum_value_dropped_on_mainline_engine():
    # cache-type-k is engine="any", so the ik-flag engine skip doesn't cover it,
    # but q6_0 is an ik-only KV-cache quant. A JSON-leftover mainline profile
    # carrying it must not emit --cache-type-k q6_0 (mainline rejects q6_0).
    argv = build_command(_srv(**{"cache-type-k": "q6_0", "cache-type-v": "q8_KV"}))
    assert "--cache-type-k" not in argv
    assert "--cache-type-v" not in argv
    assert "q6_0" not in argv
    assert "q8_KV" not in argv


def test_ik_kv_enum_value_emitted_on_ik_engine():
    # The drop is engine-scoped, not a blanket ban: on the ik engine the same
    # value renders normally.
    p = _ik_profile("ik_llama.cpp")
    p.settings["cache-type-k"] = "q6_0"
    argv = build_command(p)
    assert argv[argv.index("--cache-type-k") + 1] == "q6_0"


def test_mainline_kv_quant_still_emitted():
    # A mainline KV quant (q8_0) is not ik-only and must keep emitting.
    argv = build_command(_srv(**{"cache-type-k": "q8_0"}))
    assert argv[argv.index("--cache-type-k") + 1] == "q8_0"


def test_mla_use_auto_sentinel_not_emitted():
    # mla-use default "auto" is a non-emit sentinel (ik wants an int 0-3);
    # a JSON-leftover value of "auto" must not reach argv as --mla-use auto.
    p = _ik_profile("ik_llama.cpp")
    p.settings["mla-use"] = "auto"
    argv = build_command(p)
    assert "--mla-use" not in argv


def test_mla_use_real_value_emitted():
    # Only the sentinel is dropped; a real level still renders.
    p = _ik_profile("ik_llama.cpp")
    p.settings["mla-use"] = "2"
    argv = build_command(p)
    assert argv[argv.index("--mla-use") + 1] == "2"


def test_native_launch_uses_binary_not_container():
    p = Profile(name="n", model="/home/me/models/m.gguf",
                runtime=Runtime(launch_mode="native",
                                native_binary="/opt/bin/llama-server",
                                bind_host="127.0.0.1"),
                settings={"port": 9001})
    argv = build_command(p)
    assert argv[0] == "/opt/bin/llama-server"
    assert "run" not in argv and "-v" not in argv and "--name" not in argv
    assert "-m" in argv and "/home/me/models/m.gguf" in argv
    # binds directly to the profile's host (no -p publish in native mode)
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "9001"


def test_native_host_defaults_to_bind_host_0000():
    p = Profile(name="n", model="/m.gguf",
                runtime=Runtime(launch_mode="native",
                                native_binary="/opt/bin/llama-server",
                                bind_host="0.0.0.0"),
                settings={"port": 8080})
    argv = build_command(p)
    assert argv[argv.index("--host") + 1] == "0.0.0.0"


def test_container_launch_host_unchanged():
    # Container mode still binds 0.0.0.0 inside the container regardless of bind_host.
    p = Profile(name="c", image="img:tag", model="/models/m.gguf",
                runtime=Runtime(bind_host="127.0.0.1"), settings={"port": 8080})
    argv = build_command(p)
    assert argv[argv.index("--host") + 1] == "0.0.0.0"
    assert argv[0] == "podman" and "run" in argv


# -- ik_llama.cpp layer tokens ------------------------------------------------
# ik parses -ngl/-ngld with a bare stoi: "--n-gpu-layers all" and "auto" print
# "stoi" plus the usage text and exit before the model loads, while an
# integer proceeds.

def test_ik_launch_translates_all_layers_to_an_integer():
    p = _ik_srv(**{"n-gpu-layers": "all", "spec-draft-ngl": "all"})
    p.draft_model = "/models/draft.gguf"
    argv = build_command(p)
    assert argv[argv.index("--n-gpu-layers") + 1] == "999"
    assert argv[argv.index("--gpu-layers-draft") + 1] == "999"
    assert "all" not in argv


def test_ik_launch_drops_auto_layers_instead_of_emitting_a_token():
    p = _ik_srv(**{"n-gpu-layers": "auto", "spec-draft-ngl": "auto"})
    argv = build_command(p)
    assert "--n-gpu-layers" not in argv
    assert "--gpu-layers-draft" not in argv
    assert "auto" not in argv


def test_ik_launch_passes_integer_layers_through():
    p = _ik_srv(**{"n-gpu-layers": 40})
    argv = build_command(p)
    assert argv[argv.index("--n-gpu-layers") + 1] == "40"


def test_mainline_launch_keeps_layer_tokens():
    # Mainline accepts the tokens; the translation is ik-only.
    p = _golden_profile()
    p.settings["n-gpu-layers"] = "auto"
    argv = build_command(p)
    assert argv[argv.index("--n-gpu-layers") + 1] == "auto"
