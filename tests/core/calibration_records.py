"""Measured runs the memory estimate is calibrated against.

Each record carries the model's header block, the profile settings that
shaped the run, the free VRAM per card at launch and the buffer sizes the
engine logged, all in bytes. Records are read by the calibration test and by
scripts/fit_compute_terms.py; nothing under src reads them. A record may also
carry a `draft_meta` header for a run that loaded a draft model, and may set
`pending` to a tuple naming any of "compute", "host" or "output" to keep
those figures out of their bands while every figure it does not name, and
the KV and state bands, still apply.
"""

from types import SimpleNamespace

from llama_launcher.core import vram
from llama_launcher.core.gguf import TensorInfo

MIB = 1024 * 1024


def estimate_for(record):
    """The record's estimate, with one byte of weights standing for the
    model so placement runs on the record's settings alone, and the same for
    a draft the record carries."""
    meta = SimpleNamespace(**record["meta"])
    draft = record.get("draft_meta")
    return vram.estimate_memory(
        meta,
        1,
        settings=record["settings"],
        engine=record["engine"],
        free_bytes_per_gpu=record["free_bytes_per_gpu"],
        draft_meta=SimpleNamespace(**draft) if draft else None,
        draft_weights=1 if draft else 0,
    )


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
    head_dim_k_swa=None,
    head_dim_v_swa=None,
    sliding_window_pattern=None,
    shared_kv_layers=None,
    tensors=(),
)

# Qwen3.8-27B UD-Q4_K_S, arch qwen35: one header block, its 65 block_count
# positions covering 64 real hybrid layers plus one multi-token-prediction
# tail position (nextn_predict_layers 1) that holds neither a KV cache nor
# recurrent state, whether or not a draft loads.
_QWEN_27B = dict(
    n_layers=65,
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
    head_dim_k_swa=None,
    head_dim_v_swa=None,
    sliding_window_pattern=None,
    shared_kv_layers=None,
    tensors=(),
)

# A stand-in for the draft file's own tensor table: the four blk.64.nextn.*
# names plus an attention pair, none of them logged (the draft's eighteen
# tensor names never appear in the main file's log), chosen only so
# layer_index resolves every one of them to block 64.
_QWEN_27B_DRAFT_TENSORS = tuple(
    TensorInfo(f"blk.64.{name}.weight", 1, 0, 1)
    for name in (
        "nextn.eh_proj",
        "nextn.enorm",
        "nextn.hnorm",
        "nextn.shared_head_norm",
        "attn_q",
        "attn_k",
    )
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
            head_dim_k_swa=256,
            head_dim_v_swa=256,
            sliding_window_pattern=(True, True, True, True, False) * 7,
            shared_kv_layers=20,
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
    },
    {
        "name": "qwen3.8-27B q4_k_s mainline sweep n-cpu-ffn 0",
        "engine": "llama.cpp",
        "meta": dict(_QWEN_27B),
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
    },
    {
        # Gemma 4 12B UD-Q4_K_XL, mainline b10818, two cards by free-memory
        # proportions, --swa-full, one slot. The run also loaded an MTP draft
        # model; its buffers are left out, since the estimate has no term for
        # a draft's compute buffer and the main model's own lines split off
        # cleanly.
        "name": "gemma-4-12B ud-q4_k_xl mainline swa-full",
        "engine": "llama.cpp",
        "meta": dict(
            n_layers=48,
            n_head=16,
            n_head_kv=8,
            n_embd=3840,
            n_ff=15360,
            n_ff_exp=None,
            n_expert_used=None,
            expert_count=None,
            n_vocab=262144,
            sliding_window=1024,
            head_dim_k=512,
            head_dim_v=512,
            head_dim_k_swa=256,
            head_dim_v_swa=256,
            sliding_window_pattern=(True, True, True, True, True, False) * 8,
            shared_kv_layers=0,
            full_attention_interval=None,
            kv_layer_heads=(8, 8, 8, 8, 8, 1) * 8,
            ssm_conv_kernel=None,
            ssm_inner_size=None,
            ssm_state_size=None,
            ssm_group_count=None,
            nextn_predict_layers=None,
            ctx_train=262144,
            tensors=(),
        ),
        "settings": {
            "ctx-size": 32768,
            "swa-full": True,
            "n-gpu-layers": "all",
            "no-mmproj-offload": True,
            "threads": 12,
            "parallel": 1,
            "verbosity": 4,
        },
        "free_bytes_per_gpu": [14558429184, 12448694272],
        "measured": {
            "cards": [
                {
                    "model": int(3285.98 * MIB),
                    "kv": (256 + 5888) * MIB,
                    "compute": int(407.07 * MIB),
                },
                {
                    "model": int(3104.21 * MIB),
                    "kv": (256 + 4352) * MIB,
                    "compute": int(407.07 * MIB),
                },
            ],
            "ram": {
                "model": 540 * MIB,
                "kv": 0,
                "compute": int(271.08 * MIB),
                "output": 1 * MIB,
            },
        },
    },
    {
        # Gemma 4 26B-A4B MXFP4_MOE, mainline b10818, two cards by free-memory
        # proportions, --swa-full, one slot. The expert width comes from the
        # loader's metadata dump (gemma4.expert_feed_forward_length 704).
        "name": "gemma-4-26B-A4B mxfp4 mainline swa-full",
        "engine": "llama.cpp",
        "meta": dict(
            n_layers=30,
            n_head=16,
            n_head_kv=8,
            n_embd=2816,
            n_ff=2112,
            n_ff_exp=704,
            n_expert_used=8,
            expert_count=None,
            n_vocab=262144,
            sliding_window=1024,
            head_dim_k=512,
            head_dim_v=512,
            head_dim_k_swa=256,
            head_dim_v_swa=256,
            sliding_window_pattern=(True, True, True, True, True, False) * 5,
            shared_kv_layers=0,
            full_attention_interval=None,
            kv_layer_heads=(8, 8, 8, 8, 8, 2) * 5,
            ssm_conv_kernel=None,
            ssm_inner_size=None,
            ssm_state_size=None,
            ssm_group_count=None,
            nextn_predict_layers=None,
            ctx_train=262144,
            tensors=(),
        ),
        "settings": {
            "ctx-size": 32768,
            "swa-full": True,
            "n-gpu-layers": "all",
            "no-mmproj-offload": True,
            "threads": 12,
            "parallel": 1,
            "verbosity": 4,
        },
        "free_bytes_per_gpu": [14563672064, 12448694272],
        "measured": {
            "cards": [
                {
                    "model": int(8414.18 * MIB),
                    "kv": (256 + 3840) * MIB,
                    "compute": int(412.57 * MIB),
                },
                {
                    "model": int(7355.12 * MIB),
                    "kv": (384 + 2560) * MIB,
                    "compute": int(412.57 * MIB),
                },
            ],
            "ram": {
                "model": 748 * MIB,
                "kv": 0,
                "compute": int(267.08 * MIB),
                "output": 1 * MIB,
            },
        },
    },
    {
        # Qwen3.8-27B UD-Q4_K_S, mainline b10818, one slot, MTP draft on at
        # --spec-draft-n-max 2, tensor-split 43,23, KV q8_0. Card 1's model
        # buffer is the main model's 5883.07 MiB plus the draft's 774.71 MiB.
        # The estimate does not yet reproduce the output buffer: the draft
        # model logs its own output buffer alongside the main model's, and
        # the estimate carries only one.
        "name": "qwen3.8-27B q4_k_s mainline mtp draft on",
        "engine": "llama.cpp",
        "meta": dict(_QWEN_27B),
        "draft_meta": dict(_QWEN_27B, tensors=_QWEN_27B_DRAFT_TENSORS),
        "pending": ("output",),
        "settings": {
            "ctx-size": 90112,
            "tensor-split": "43,23",
            "cache-type-k": "q8_0",
            "cache-type-v": "q8_0",
            "flash-attn": "on",
            "parallel": 1,
            "spec-type": "draft-mtp",
            "spec-draft-n-max": 2,
            "verbosity": 4,
        },
        "free_bytes_per_gpu": [int(14897 * MIB), int(11768 * MIB)],
        "measured": {
            "cards": [
                {
                    "model": int(8232.19 * MIB),
                    "kv": int((1870.00 + 308.60) * MIB),
                    "compute": int(825.13 * MIB),
                },
                {
                    "model": int((5883.07 + 774.71) * MIB),
                    # main cache 1122.00, draft cache 352.00, recurrent state 140.27
                    "kv": int((1122.00 + 352.00 + 140.27) * MIB),
                    "compute": int((825.13 + 554.06) * MIB),
                },
            ],
            "ram": {
                "model": int((521.00 + 521.00) * MIB),
                "kv": 0,
                "compute": int((373.13 + 402.07) * MIB),
                "output": int(0.95 * MIB) + int(0.95 * MIB),
            },
        },
    },
    {
        # Qwen3.6-35B-A3B MXFP4, mainline b10818, four slots, no draft,
        # kv_unified true. The control run for the 27B draft record: with no
        # draft the state term lands exactly. The estimate does not yet
        # reproduce card 0's compute buffer or the host buffer: both read
        # low against the log.
        "name": "qwen3.6-35B-A3B mxfp4 mainline four slots kv-unified",
        "engine": "llama.cpp",
        "meta": dict(_QWEN_35B_A3B),
        "pending": ("compute", "host"),
        "settings": {
            "ctx-size": 131072,
            "n-gpu-layers": "all",
            "flash-attn": "on",
            "no-mmproj-offload": True,
            "threads": 12,
            "parallel": 4,
            "kv-unified": True,
            "verbosity": 4,
        },
        "free_bytes_per_gpu": [int(14767 * MIB), int(11768 * MIB)],
        "measured": {
            "cards": [
                {
                    "model": int(11780.09 * MIB),
                    "kv": int((1536.00 + 150.75) * MIB),
                    "compute": int(628.10 * MIB),
                },
                {
                    "model": int(8394.71 * MIB),
                    "kv": int((1024.00 + 100.50) * MIB),
                    "compute": int(628.10 * MIB),
                },
            ],
            "ram": {
                "model": int(515.31 * MIB),
                "kv": 0,
                "compute": int(520.07 * MIB),
                "output": int(3.79 * MIB),
            },
        },
    },
    {
        # Qwen3.8-27B UD-Q4_K_S again, mainline b10818, two slots, MTP off:
        # ctx 90112 divided into two slots of 45056. The main file's MTP
        # tensors (blk.64.nextn.*) go unused, so card 1's model buffer drops
        # from 5883.07 to 5548.32 MiB. The estimate does not yet reproduce
        # the host buffer: it reads several times high against the log.
        "name": "qwen3.8-27B q4_k_s mainline two slots mtp off",
        "engine": "llama.cpp",
        "meta": dict(_QWEN_27B),
        "pending": ("host",),
        "settings": {
            "ctx-size": 90112,
            "tensor-split": "43,23",
            "cache-type-k": "q8_0",
            "cache-type-v": "q8_0",
            "flash-attn": "on",
            "parallel": 2,
            "verbosity": 4,
        },
        "free_bytes_per_gpu": [int(14771 * MIB), int(11768 * MIB)],
        "measured": {
            "cards": [
                {
                    "model": int(8232.19 * MIB),
                    "kv": int((1870.00 + 205.73) * MIB),
                    "compute": int(649.13 * MIB),
                },
                {
                    "model": int(5548.32 * MIB),
                    "kv": int((1122.00 + 93.52) * MIB),
                    "compute": int(649.13 * MIB),
                },
            ],
            "ram": {
                "model": int(521.00 * MIB),
                "kv": 0,
                "compute": int(197.13 * MIB),
                "output": int(1.89 * MIB),
            },
        },
    },
    {
        # Qwen3.8-27B UD-Q4_K_S, mainline b10818, two slots, MTP draft on at
        # --spec-draft-n-max 2, tensor-split 43,23, KV q8_0: two slots at
        # depth 2 charge 6 state units per layer, which a one-slot run
        # cannot distinguish from 3. The state logs 897.75 MiB in total,
        # 2 cells, 2 seqs, 2 rs_seq over the 48 recurrent layers of 64,
        # that is 6 units per layer; per card 617.20 (33 layers) and
        # 280.55 (15). Card 1's model buffer is the main model's 5883.07
        # MiB plus the draft's 774.71 MiB; the draft's cache reads 352.00
        # MiB at 45056 cells times 2/2 seqs, the same buffer as at one
        # slot. The estimate carries one output buffer where this run
        # logs two, one per model.
        "name": "qwen3.8-27B q4_k_s mainline two slots mtp draft on",
        "engine": "llama.cpp",
        "meta": dict(_QWEN_27B),
        "draft_meta": dict(_QWEN_27B, tensors=_QWEN_27B_DRAFT_TENSORS),
        "pending": ("output",),
        "settings": {
            "ctx-size": 90112,
            "tensor-split": "43,23",
            "cache-type-k": "q8_0",
            "cache-type-v": "q8_0",
            "flash-attn": "on",
            "parallel": 2,
            "spec-type": "draft-mtp",
            "spec-draft-n-max": 2,
            "verbosity": 4,
        },
        "free_bytes_per_gpu": [int(14813 * MIB), int(11768 * MIB)],
        "measured": {
            "cards": [
                {
                    "model": int(8232.19 * MIB),
                    "kv": int((1870.00 + 617.20) * MIB),
                    "compute": int(649.13 * MIB),
                },
                {
                    "model": int((5883.07 + 774.71) * MIB),
                    # main cache 1122.00, draft cache 352.00, recurrent state 280.55
                    "kv": int((1122.00 + 352.00 + 280.55) * MIB),
                    "compute": int((649.13 + 378.06) * MIB),
                },
            ],
            "ram": {
                "model": int((521.00 + 521.00) * MIB),
                "kv": 0,
                "compute": int((197.13 + 226.07) * MIB),
                "output": int(1.89 * MIB) + int(1.89 * MIB),
            },
        },
    },
]
