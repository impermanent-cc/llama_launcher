from llama_launcher.core.settings_catalog import CATALOG, Setting, KV_CACHE_TYPES


def test_catalog_keys_match_their_setting_key():
    for key, s in CATALOG.items():
        assert s.key == key
        assert s.flag.startswith("--")


def test_known_settings_present_with_correct_defaults():
    assert CATALOG["ctx-size"].default == 0
    assert CATALOG["flash-attn"].type == "enum"
    assert CATALOG["flash-attn"].enum == ("on", "off", "auto")
    assert CATALOG["flash-attn"].default == "auto"
    assert CATALOG["repeat-penalty"].default == 1.0
    assert CATALOG["cache-type-k"].enum == KV_CACHE_TYPES
    assert CATALOG["tools"].danger is True
    assert CATALOG["n-gpu-layers"].type == "int_or_token"
    assert "auto" in CATALOG["n-gpu-layers"].tokens
    assert "all" in CATALOG["n-gpu-layers"].tokens


def test_no_deprecated_defrag_thold():
    assert "defrag-thold" not in CATALOG


def test_enum_defaults_are_within_enum():
    for s in CATALOG.values():
        if s.type == "enum":
            assert s.default in s.enum


def test_speculative_decoding_group():
    from llama_launcher.core.settings_catalog import CATALOG, KV_CACHE_TYPES
    assert CATALOG["spec-type"].type == "enum"
    assert CATALOG["spec-type"].default == "none"
    assert "draft-mtp" in CATALOG["spec-type"].enum
    # Full llama.cpp strategy set (build b9755): the two draft-d* variants were
    # missing from the enum, so the UI couldn't select them.
    assert "draft-dflash" in CATALOG["spec-type"].enum
    assert "draft-dspark" in CATALOG["spec-type"].enum
    assert CATALOG["spec-draft-n-min"].default == 0
    assert CATALOG["cache-type-k-draft"].enum == KV_CACHE_TYPES
    assert CATALOG["cache-type-v-draft"].default == "f16"
    for k in ("spec-type", "spec-draft-ngl", "spec-draft-n-max",
              "spec-draft-n-min", "cache-type-k-draft", "cache-type-v-draft"):
        assert CATALOG[k].group == "Speculative Decoding", k


def test_chat_template_kwargs_present():
    s = CATALOG["chat-template-kwargs"]
    assert s.flag == "--chat-template-kwargs"
    assert s.type == "string"
    assert s.default == ""
    assert s.group == "Server & Tools"


def test_reasoning_budget_message_present():
    s = CATALOG["reasoning-budget-message"]
    assert s.flag == "--reasoning-budget-message"
    assert s.type == "string"
    assert s.default == ""
    assert s.group == "Server & Tools"


def test_spec_draft_backend_sampling_present_as_negatable_bool():
    # Upstream default is enabled; represented like cors-credentials -- a bool
    # defaulting False whose flag is the --no- form, so checking it emits the
    # disable flag and leaving it unchecked emits nothing.
    s = CATALOG["spec-draft-backend-sampling"]
    assert s.flag == "--no-spec-draft-backend-sampling"
    assert s.type == "bool"
    assert s.default is False
    assert s.group == "Speculative Decoding"


def test_gpu_additions_and_split_mode_tensor():
    from llama_launcher.core.settings_catalog import CATALOG
    assert CATALOG["no-mmproj-offload"].type == "bool"
    assert CATALOG["no-mmproj-offload"].default is False
    assert CATALOG["override-tensor"].type == "string"
    assert "-ot" in CATALOG["override-tensor"].aliases
    assert CATALOG["split-mode"].enum == ("none", "layer", "row", "tensor")


def test_legacy_load_flags_are_marked_deprecated():
    # Upstream (v0.2.0 era) now emits DEPRECATED warnings for these and steers
    # users to --load-mode; we keep them for older images but flag them.
    assert CATALOG["no-mmap"].deprecated is True
    assert CATALOG["mlock"].deprecated is True
    # --load-mode is the replacement and must NOT be marked deprecated.
    assert CATALOG["load-mode"].deprecated is False


def test_most_settings_are_not_deprecated():
    deprecated = [k for k, s in CATALOG.items() if s.deprecated]
    assert set(deprecated) == {"no-mmap", "mlock"}


def test_mmproj_device_present():
    # New llama-server flag in the v0.1.x -> v0.2.0 window: pick the device the
    # multimodal projector runs on.
    s = CATALOG["mmproj-device"]
    assert s.flag == "--mmproj-device"
    assert s.type == "string"
    assert s.default == ""
    assert s.group == "GPU & Memory"
    assert "-mmdev" in s.aliases


def test_context_and_caching_additions():
    from llama_launcher.core.settings_catalog import CATALOG
    assert CATALOG["swa-full"].type == "bool"
    assert CATALOG["swa-full"].group == "Model & Context"
    assert CATALOG["context-shift"].type == "bool"
    assert CATALOG["ctx-checkpoints"].default == 32
    assert CATALOG["ctx-checkpoints"].group == "Caching"
    assert CATALOG["checkpoint-min-step"].default == 8192
    assert "-ctxcp" in CATALOG["ctx-checkpoints"].aliases
    assert "-cms" in CATALOG["checkpoint-min-step"].aliases


def test_sampling_perf_server_additions():
    from llama_launcher.core.settings_catalog import CATALOG
    assert CATALOG["dry-sequence-breaker"].type == "string"
    assert CATALOG["dry-sequence-breaker"].group == "Sampling"
    assert CATALOG["numa"].type == "enum"
    assert CATALOG["numa"].default == "off"
    assert CATALOG["numa"].enum == ("off", "distribute", "isolate", "numactl")
    assert CATALOG["threads-http"].default == -1
    assert CATALOG["no-webui"].type == "bool"
    assert CATALOG["reasoning-format"].default == "auto"
    assert CATALOG["reasoning-format"].enum == ("auto", "none", "deepseek", "deepseek-legacy")


def test_checkpoint_min_step_matches_upstream_default():
    # llama.cpp b10290 changed this default from 256 to 8192. A stale default
    # means the widget shows 256, emits nothing, and the server silently uses 8192.
    assert CATALOG["checkpoint-min-step"].default == 8192


def test_dry_penalty_last_n_matches_upstream_default():
    # b10290: default -1 -> 64, and -1 no longer means "context size".
    s = CATALOG["dry-penalty-last-n"]
    assert s.default == 64
    assert s.minimum == 0


def test_repeat_last_n_no_longer_offers_dead_sentinel():
    # b10290 dropped "-1 = ctx_size" for repeat-last-n.
    s = CATALOG["repeat-last-n"]
    assert s.default == 64
    assert s.minimum == 0


from llama_launcher.core.settings_catalog import (
    HOST_KEYS, ROUTER_ONLY_KEYS, member_catalog, router_catalog,
)


def test_router_settings_present():
    assert CATALOG["models-max"].flag == "--models-max"
    assert CATALOG["models-max"].default == 4          # mirrors upstream; see the test below
    assert CATALOG["models-autoload"].type == "bool"
    assert CATALOG["sleep-idle-seconds"].flag == "--sleep-idle-seconds"
    assert CATALOG["sleep-idle-seconds"].default == -1  # -1 = disabled


def test_ride_along_settings_present():
    for key, flag in [
        ("cors-origins", "--cors-origins"),
        ("cors-methods", "--cors-methods"),
        ("cors-headers", "--cors-headers"),
        ("cors-credentials", "--no-cors-credentials"),
        ("sse-ping-interval", "--sse-ping-interval"),
        ("mcp-servers-config", "--mcp-servers-config"),
        ("mcp-servers-json", "--mcp-servers-json"),
        ("reasoning-preserve", "--reasoning-preserve"),
    ]:
        assert CATALOG[key].flag == flag


def test_host_and_router_key_sets_reference_real_settings():
    # A typo in either set would silently drop a setting from a form.
    assert HOST_KEYS <= set(CATALOG)
    assert ROUTER_ONLY_KEYS <= HOST_KEYS


def test_router_catalog_excludes_model_level_settings():
    rc = router_catalog()
    # The precedence trap: router CLI args OVERRIDE per-model preset values, so
    # model-level settings must not be settable on a router profile.
    for key in ("ctx-size", "n-gpu-layers", "temp", "spec-type", "n-cpu-moe"):
        assert key not in rc
    assert "models-max" in rc
    assert "port" in rc


def test_member_catalog_excludes_router_only_settings():
    mc = member_catalog()
    assert "models-max" not in mc
    assert "models-autoload" not in mc
    assert "ctx-size" in mc
    assert "sleep-idle-seconds" in mc  # valid in single-model mode too


def test_models_max_default_matches_upstream_not_advice():
    # Widgets emit only when value != default, so a default of 1 (the *advice*)
    # meant leaving the field alone emitted nothing and the server ran 4.
    assert CATALOG["models-max"].default == 4


def test_models_autoload_uses_the_negative_flag():
    # A bool with default True can never emit: checked == default (nothing
    # stored), unchecked renders [] because _render_setting skips falsy bools.
    # The controllable form is the negative one, as with cors-credentials.
    s = CATALOG["models-autoload"]
    assert s.flag == "--no-models-autoload"
    assert s.default is False


def test_load_mode_setting():
    from llama_launcher.core.settings_catalog import CATALOG
    s = CATALOG["load-mode"]
    assert s.flag == "--load-mode"
    assert s.type == "enum"
    assert s.default == "mmap"
    assert s.enum == ("mmap", "none", "mlock", "mmap+mlock", "dio")
    assert "-lm" in s.aliases
    # legacy bools retained for older images
    assert CATALOG["no-mmap"].type == "bool"
    assert CATALOG["mlock"].type == "bool"


from llama_launcher.core.settings_catalog import (
    CATALOG, member_catalog, for_engine, IK_EXTRA_KV_CACHE_TYPES,
)

_IK_KEYS = {"run-time-repack", "no-fused-moe", "mla-use",
            "attention-max-batch", "smart-expert-reduction"}


def test_ik_flags_exist_and_are_engine_tagged():
    for k in _IK_KEYS:
        assert k in CATALOG, k
        assert CATALOG[k].engine == "ik_llama.cpp", k
        assert CATALOG[k].group == "ik_llama.cpp", k


def test_existing_settings_default_to_engine_any():
    assert CATALOG["ctx-size"].engine == "any"


def test_for_engine_drops_ik_flags_for_llama_cpp():
    cat = for_engine(member_catalog(), "llama.cpp")
    assert _IK_KEYS.isdisjoint(cat)
    # a plain setting still present
    assert "ctx-size" in cat


def test_for_engine_keeps_ik_flags_for_ik():
    cat = for_engine(member_catalog(), "ik_llama.cpp")
    assert _IK_KEYS <= set(cat)


def test_ik_extra_kv_cache_types_are_additive():
    from llama_launcher.core.settings_catalog import KV_CACHE_TYPES
    assert set(IK_EXTRA_KV_CACHE_TYPES).isdisjoint(KV_CACHE_TYPES)
    assert "q6_0" in IK_EXTRA_KV_CACHE_TYPES
