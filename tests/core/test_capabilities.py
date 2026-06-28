from llama_launcher.core.gguf import GgufMeta
from llama_launcher.core.capabilities import derive_caps, Tier, relevance, suggestions


def test_qwen_mtp_infile():
    meta = GgufMeta(arch="qwen35moe", expert_count=256, nextn_predict_layers=1, ctx_train=262144)
    caps = derive_caps(meta, [])
    assert caps.is_moe and caps.expert_count == 256
    assert caps.has_mtp_infile and caps.has_mtp
    assert caps.mtp_sibling is None and not caps.has_vision


def test_gemma_companion_siblings():
    meta = GgufMeta(arch="gemma4", sliding_window=1024, ctx_train=262144)
    caps = derive_caps(meta, ["mtp-gemma-4-12B-it.gguf", "mmproj-F16.gguf"])
    assert not caps.is_moe and caps.has_swa
    assert caps.mtp_sibling == "mtp-gemma-4-12B-it.gguf" and caps.has_mtp
    assert caps.mmproj_sibling == "mmproj-F16.gguf" and caps.has_vision


def test_plain_model_has_no_mtp_or_vision():
    caps = derive_caps(GgufMeta(arch="qwen35moe", expert_count=256, ctx_train=262144), [])
    assert caps.is_moe and not caps.has_mtp and not caps.has_vision


def test_none_meta_is_empty_caps():
    caps = derive_caps(None, [])
    assert not caps.is_moe and not caps.has_mtp and caps.ctx_train is None


def test_relevance_moe_vs_dense():
    moe = relevance(derive_caps(GgufMeta(expert_count=256), []))
    assert moe["n-cpu-moe"] == Tier.RECOMMENDED
    dense = relevance(derive_caps(GgufMeta(), []))
    assert dense["n-cpu-moe"] == Tier.NA


def test_relevance_mtp_vs_none():
    mtp = relevance(derive_caps(GgufMeta(nextn_predict_layers=1), []))
    assert mtp["spec-type"] == Tier.RECOMMENDED
    plain = relevance(derive_caps(GgufMeta(), []))
    assert plain["spec-type"] == Tier.NA


def test_relevance_vision_and_baseline():
    vis = relevance(derive_caps(GgufMeta(), ["mmproj-F16.gguf"]))
    assert vis["mmproj"] == Tier.RECOMMENDED
    novis = relevance(derive_caps(GgufMeta(), []))
    assert novis["mmproj"] == Tier.NA
    assert novis["ctx-size"] == Tier.RECOMMENDED      # baseline always
    assert novis["threads"] == Tier.TUNE


def test_suggestion_mtp_infile():
    caps = derive_caps(GgufMeta(nextn_predict_layers=1), [])
    sg = suggestions(caps, {"spec-type": "none"}, mmproj_set=False, draft_set=False)
    assert any(s.settings.get("spec-type") == "draft-mtp" for s in sg)
    # suppressed once already set
    assert suggestions(caps, {"spec-type": "draft-mtp"}, False, False) == []


def test_suggestion_sibling_files_carry_filenames():
    caps = derive_caps(GgufMeta(), ["mtp-g.gguf", "mmproj-F16.gguf"])
    sg = suggestions(caps, {}, mmproj_set=False, draft_set=False)
    fields = {k: v for s in sg for k, v in s.fields.items()}
    assert fields.get("draft_model") == "mtp-g.gguf"
    assert fields.get("mmproj") == "mmproj-F16.gguf"


def test_suggestion_ctx_cap():
    caps = derive_caps(GgufMeta(ctx_train=262144), [])
    sg = suggestions(caps, {"ctx-size": 999999}, False, False)
    assert any(s.settings.get("ctx-size") == 262144 for s in sg)
