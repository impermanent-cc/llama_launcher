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
