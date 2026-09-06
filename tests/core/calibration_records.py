"""Measured runs the memory estimate is calibrated against.

Each record carries the model's header block, the profile settings that
shaped the run, the free VRAM per card at launch and the buffer sizes the
engine logged, all in bytes. `output_card` is the card whose measured
compute buffer carries the output layer, or None when the output layer
stayed in host RAM. Records are read by the calibration test and by
scripts/fit_compute_terms.py; nothing under src reads them.
"""

MIB = 1024 * 1024

# Qwen3.6-35B-A3B MXFP4: one header block, two engines. The expert width
# n_ff_exp is read from the ik launch log's metadata dump, key 29,
# qwen35moe.expert_feed_forward_length u32 = 512.
_QWEN_35B_A3B = dict(
    n_layers=40,
    n_head=16,
    n_head_kv=2,
    n_embd=2048,
    n_ff=0,
    n_ff_exp=512,
    n_expert_used=8,
    expert_count=256,
    n_vocab=248320,
    sliding_window=None,
    head_dim_k=256,
    head_dim_v=256,
    full_attention_interval=4,
    kv_layer_heads=None,
    ssm_conv_kernel=4,
    ssm_inner_size=4096,
    ssm_state_size=128,
    ssm_group_count=16,
    nextn_predict_layers=None,
    ctx_train=None,
    tensors=(),
)

RECORDS = [
    {
        "name": "gemma-4-E2B mainline cpu-only",
        "engine": "llama.cpp",
        "meta": dict(
            n_layers=35,
            n_head=8,
            n_head_kv=1,
            n_embd=1536,
            n_ff=12288,
            n_ff_exp=None,
            n_expert_used=None,
            expert_count=None,
            n_vocab=262144,
            sliding_window=512,
            head_dim_k=512,
            head_dim_v=512,
            full_attention_interval=None,
            kv_layer_heads=None,
            ssm_conv_kernel=None,
            ssm_inner_size=None,
            ssm_state_size=None,
            ssm_group_count=None,
            nextn_predict_layers=None,
            ctx_train=131072,
            tensors=(),
        ),
        "settings": {
            "ctx-size": 2048,
            "n-gpu-layers": 0,
            "flash-attn": "on",
            "threads": 6,
        },
        "free_bytes_per_gpu": [],
        "measured": {
            "cards": [],
            "ram": {
                "model": int(2483.69 * MIB) + int(1222.80 * MIB),
                "kv": 36 * MIB,
                "compute": int(113.52 * MIB),
                "output": 4 * MIB,
            },
        },
        "output_card": None,
    },
    {
        "name": "qwen3.6-35B-A3B mxfp4 ik_llama.cpp fit",
        "engine": "ik_llama.cpp",
        "meta": dict(_QWEN_35B_A3B),
        "settings": {
            "ctx-size": 131072,
            "n-gpu-layers": "all",
            "flash-attn": "on",
            "no-mmproj-offload": True,
            "threads": 12,
            "fit": "on",
            "verbosity": 4,
        },
        "free_bytes_per_gpu": [14743 * MIB, 11768 * MIB],
        "measured": {
            "cards": [
                {
                    "model": int(11780.09 * MIB),
                    "kv": int(1573.69 * MIB),
                    "compute": int(194.17 * MIB),
                },
                {
                    "model": int(8394.71 * MIB),
                    "kv": int(1049.12 * MIB),
                    "compute": int(489.00 * MIB),
                },
            ],
            "ram": {
                "model": int(515.31 * MIB),
                "kv": 0,
                "compute": int(132.01 * MIB),
                "output": int(0.95 * MIB),
            },
        },
        "output_card": 1,
    },
    {
        "name": "qwen3.6-35B-A3B mxfp4 mainline sweep n-cpu-moe 2",
        "engine": "llama.cpp",
        "meta": dict(_QWEN_35B_A3B),
        "settings": {
            "ctx-size": 131072,
            "n-gpu-layers": "all",
            "flash-attn": "on",
            "no-mmproj-offload": True,
            "threads": 12,
            "n-cpu-moe": 2,
        },
        "free_bytes_per_gpu": [15381561344, 12448694272],
        "measured": {
            "cards": [
                {"model": 11293257891, "kv": 1610612736, "compute": 434131435},
                {"model": 8802491432, "kv": 1073741824, "compute": 239127756},
            ],
            "ram": {
                "model": 1599403458,
                "kv": 0,
                "compute": 402779012,
                "output": 3974103,
            },
        },
        "output_card": 0,
    },
    {
        "name": "qwen3.5-27B q4_k_s mainline sweep n-cpu-ffn 0",
        "engine": "llama.cpp",
        "meta": dict(
            n_layers=64,
            n_head=24,
            n_head_kv=4,
            n_embd=5120,
            n_ff=17408,
            n_ff_exp=None,
            n_expert_used=None,
            expert_count=None,
            n_vocab=248320,
            sliding_window=None,
            head_dim_k=256,
            head_dim_v=256,
            full_attention_interval=4,
            kv_layer_heads=None,
            ssm_conv_kernel=4,
            ssm_inner_size=6144,
            ssm_state_size=128,
            ssm_group_count=16,
            nextn_predict_layers=1,
            ctx_train=None,
            tensors=(),
        ),
        "settings": {
            "ctx-size": 32768,
            "tensor-split": "60,40",
            "cache-type-k": "q8_0",
            "cache-type-v": "q8_0",
            "flash-attn": "on",
            # The measured output buffer is twice vocabulary times four bytes,
            # so the run served two slots.
            "parallel": 2,
        },
        "free_bytes_per_gpu": [15395192832, 12448694272],
        "measured": {
            "cards": [
                {"model": 8632076861, "kv": 713031680, "compute": 395449466},
                {"model": 6981188320, "kv": 562036736, "compute": 1087635454},
            ],
            "ram": {
                "model": 1092616192,
                "kv": 0,
                "compute": 529813994,
                "output": 1992294,
            },
        },
        "output_card": 1,
    },
]
