from llama_launcher.core.gguf import GgufMeta
from llama_launcher.core.capabilities import derive_caps


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
