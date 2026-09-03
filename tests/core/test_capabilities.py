from llama_launcher.core.capabilities import (
    RELEVANCE_CONTRIBUTORS,
    SUGGESTION_DETECTORS,
    Tier,
    derive_caps,
    relevance,
    suggestions,
)
from llama_launcher.core.gguf import GgufMeta


def test_qwen_mtp_infile():
    meta = GgufMeta(
        arch="qwen35moe", expert_count=256, nextn_predict_layers=1, ctx_train=262144
    )
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
    caps = derive_caps(
        GgufMeta(arch="qwen35moe", expert_count=256, ctx_train=262144), []
    )
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
    assert novis["ctx-size"] == Tier.RECOMMENDED  # baseline always
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


def test_embedding_detected_by_arch():
    caps = derive_caps(GgufMeta(arch="nomic-bert"), [])
    assert caps.is_embedding and not caps.is_reranker


def test_embedding_detected_by_pooling_kv():
    caps = derive_caps(GgufMeta(arch="bert", pooling_type=1), [])
    assert caps.is_embedding
    assert caps.pooling_type == "mean"


def test_reranker_detected_by_rank_pooling():
    caps = derive_caps(GgufMeta(arch="bert", pooling_type=4), [])
    assert caps.is_reranker and caps.is_embedding
    assert caps.pooling_type == "rank"


def test_generation_model_is_not_embedding():
    caps = derive_caps(GgufMeta(arch="qwen3", expert_count=256), [])
    assert not caps.is_embedding and not caps.is_reranker
    assert caps.pooling_type is None


def test_relevance_and_suggestions_registry_preserve_behavior():
    # MoE + in-file MTP model: exercises several contributors/detectors at once.
    caps = derive_caps(
        GgufMeta(
            arch="qwen35moe", expert_count=256, nextn_predict_layers=1, ctx_train=4096
        ),
        [],
    )
    t = relevance(caps)
    assert t["n-cpu-moe"] == Tier.RECOMMENDED
    assert t["spec-type"] == Tier.RECOMMENDED
    assert t["mmproj"] == Tier.NA
    sgs = suggestions(caps, {"ctx-size": 8192}, mmproj_set=False, draft_set=False)
    texts = [s.text for s in sgs]
    assert any("draft-mtp" in x for x in texts)
    assert any("exceeds trained max" in x for x in texts)
    # registries are the real backing structure
    assert len(RELEVANCE_CONTRIBUTORS) >= 5 and len(SUGGESTION_DETECTORS) >= 3


def test_embedding_relevance_nas_sampling_and_promotes_pooling():
    t = relevance(derive_caps(GgufMeta(arch="bert", pooling_type=1), []))
    assert t["embeddings"] == Tier.RECOMMENDED
    assert t["pooling"] == Tier.RECOMMENDED
    assert t["temp"] == Tier.NA
    assert t["spec-type"] == Tier.NA
    assert t["mmproj"] == Tier.NA


def test_reranker_relevance_promotes_reranking():
    t = relevance(derive_caps(GgufMeta(arch="bert", pooling_type=4), []))
    assert t["reranking"] == Tier.RECOMMENDED


def test_embedding_suggestion_sets_flags():
    caps = derive_caps(GgufMeta(arch="nomic-bert", pooling_type=1), [])
    sgs = suggestions(caps, {})
    assert any(
        s.settings.get("embeddings") is True and s.settings.get("pooling") == "mean"
        for s in sgs
    )


def test_reranker_suggestion_sets_trio():
    caps = derive_caps(GgufMeta(arch="bert", pooling_type=4), [])
    s = next(x for x in suggestions(caps, {}) if x.settings.get("reranking"))
    assert s.settings == {"reranking": True, "pooling": "rank", "embeddings": True}


def test_no_embedding_suggestion_for_generation_model():
    caps = derive_caps(GgufMeta(arch="qwen3"), [])
    assert not any(s.settings.get("embeddings") for s in suggestions(caps, {}))
