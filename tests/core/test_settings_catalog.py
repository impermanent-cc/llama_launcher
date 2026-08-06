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
    assert CATALOG["spec-draft-n-min"].default == 0
    assert CATALOG["cache-type-k-draft"].enum == KV_CACHE_TYPES
    assert CATALOG["cache-type-v-draft"].default == "f16"
    for k in ("spec-type", "spec-draft-ngl", "spec-draft-n-max",
              "spec-draft-n-min", "cache-type-k-draft", "cache-type-v-draft"):
        assert CATALOG[k].group == "Speculative Decoding", k


def test_gpu_additions_and_split_mode_tensor():
    from llama_launcher.core.settings_catalog import CATALOG
    assert CATALOG["no-mmproj-offload"].type == "bool"
    assert CATALOG["no-mmproj-offload"].default is False
    assert CATALOG["override-tensor"].type == "string"
    assert "-ot" in CATALOG["override-tensor"].aliases
    assert CATALOG["split-mode"].enum == ("none", "layer", "row", "tensor")


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
