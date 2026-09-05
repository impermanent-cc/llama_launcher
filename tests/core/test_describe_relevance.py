from llama_launcher.core.capabilities import (
    Tier,
    derive_caps,
    describe_relevance,
    relevance,
)
from llama_launcher.core.gguf import GgufMeta


def test_keys_match_relevance():
    caps = derive_caps(
        GgufMeta(
            arch="qwen35moe", expert_count=256, nextn_predict_layers=1, ctx_train=4096
        ),
        [],
    )
    assert set(describe_relevance(caps)) == set(relevance(caps))


def test_moe_reason_is_explained():
    caps = derive_caps(GgufMeta(expert_count=256), [])
    tier, reason = describe_relevance(caps)["n-cpu-moe"]
    assert tier == Tier.RECOMMENDED
    assert "MoE" in reason or "expert" in reason.lower()


def test_moe_na_key_has_not_applicable_reason():
    caps = derive_caps(GgufMeta(), [])  # dense model, no experts
    tier, reason = describe_relevance(caps)["n-cpu-moe"]
    assert tier == Tier.NA
    assert "not" in reason.lower()


def test_mtp_reason_is_explained():
    caps = derive_caps(GgufMeta(nextn_predict_layers=1), [])
    tier, reason = describe_relevance(caps)["spec-type"]
    assert tier == Tier.RECOMMENDED
    assert "mtp" in reason.lower() or "draft" in reason.lower()


def test_mtp_na_key_has_not_applicable_reason():
    caps = derive_caps(GgufMeta(), [])  # no MTP/draft head
    tier, reason = describe_relevance(caps)["spec-type"]
    assert tier == Tier.NA
    assert "not" in reason.lower()


def test_baseline_recommended_key_has_generic_reason():
    caps = derive_caps(GgufMeta(), [])
    tier, reason = describe_relevance(caps)["ctx-size"]
    assert tier == Tier.RECOMMENDED
    assert reason  # non-empty, human-readable


def test_baseline_tune_key_has_generic_reason():
    caps = derive_caps(GgufMeta(), [])
    tier, reason = describe_relevance(caps)["threads"]
    assert tier == Tier.TUNE
    assert reason  # non-empty, human-readable


def test_relevance_stays_unchanged():
    caps = derive_caps(GgufMeta(expert_count=256), [])
    plain = relevance(caps)
    described = describe_relevance(caps)
    assert plain == {k: t for k, (t, _) in described.items()}


def test_n_cpu_ffn_is_recommended_on_a_dense_model():
    caps = derive_caps(GgufMeta(arch="llama", ctx_train=4096), [])
    tier, reason = describe_relevance(caps)["n-cpu-ffn"]
    assert tier == Tier.RECOMMENDED
    assert "dense" in reason.lower()


def test_n_cpu_ffn_is_a_tuning_knob_on_a_moe_model():
    # A MoE gguf still carries dense FFN outside its expert layers, so the
    # flag applies there; it is just not the first thing to reach for.
    caps = derive_caps(GgufMeta(expert_count=256), [])
    tier, reason = describe_relevance(caps)["n-cpu-ffn"]
    assert tier == Tier.TUNE
    assert "n-cpu-moe" in reason
    assert describe_relevance(caps)["n-cpu-moe"][0] == Tier.RECOMMENDED
