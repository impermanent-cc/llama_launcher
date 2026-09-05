from dataclasses import dataclass
from enum import StrEnum

from .gguf import GgufMeta
from .settings_catalog import CATALOG

EMBEDDING_ARCHS = frozenset(
    {
        "bert",
        "nomic-bert",
        "nomic-bert-moe",
        "jina-bert-v2",
        "jina-bert",
        "gte",
        "gte-qwen2",
        "roberta",
        "xlm-roberta",
        "distilbert",
        "mpnet",
    }
)
_POOLING = {0: "none", 1: "mean", 2: "cls", 3: "last", 4: "rank"}


@dataclass(frozen=True)
class ModelCaps:
    is_moe: bool = False
    expert_count: int | None = None
    has_mtp_infile: bool = False
    mtp_sibling: str | None = None  # filename of a *mtp*.gguf sibling
    mmproj_sibling: str | None = None  # filename of an mmproj*.gguf sibling
    has_swa: bool = False
    sliding_window: int | None = None
    ctx_train: int | None = None
    is_embedding: bool = False
    is_reranker: bool = False
    pooling_type: str | None = None

    @property
    def has_mtp(self) -> bool:
        return self.has_mtp_infile or self.mtp_sibling is not None

    @property
    def has_vision(self) -> bool:
        return self.mmproj_sibling is not None


def _match(filenames, substr) -> str | None:
    for n in filenames:
        low = n.lower()
        if low.endswith(".gguf") and substr in low:
            return n
    return None


def derive_caps(meta: GgufMeta | None, sibling_filenames) -> ModelCaps:
    meta = meta or GgufMeta()
    pooling = _POOLING.get(meta.pooling_type) if meta.pooling_type is not None else None
    return ModelCaps(
        is_moe=bool(meta.expert_count and meta.expert_count > 0),
        expert_count=meta.expert_count,
        has_mtp_infile=bool(
            meta.nextn_predict_layers and meta.nextn_predict_layers > 0
        ),
        mtp_sibling=_match(sibling_filenames, "mtp"),
        mmproj_sibling=_match(sibling_filenames, "mmproj"),
        has_swa=bool(meta.sliding_window and meta.sliding_window > 0),
        sliding_window=meta.sliding_window,
        ctx_train=meta.ctx_train,
        is_embedding=(meta.arch in EMBEDDING_ARCHS or pooling is not None),
        # Reranker auto-detection relies on the GGUF advertising pooling_type=rank.
        # Some reranker GGUFs omit it entirely (e.g. bge-reranker-v2-m3 converts to
        # arch=bert, no pooling_type, general.name "Bge M3", indistinguishable
        # from a plain embedding model), so they aren't auto-flagged and the
        # reranking suggestion chips stay silent. They still WORK: the user enables
        # --reranking + --pooling rank + --embeddings manually, and validate()'s
        # bad-combo warnings guide that. Metadata-only on purpose (no filename
        # heuristic) to avoid false positives.
        is_reranker=(pooling == "rank"),
        pooling_type=pooling,
    )


class Tier(StrEnum):
    RECOMMENDED = "recommended"  # set this for this model
    TUNE = "tune"  # worth tuning
    NEUTRAL = "neutral"  # default styling (UI default; not returned)
    NA = "na"  # not applicable for this model


@dataclass(frozen=True)
class Suggestion:
    text: str
    settings: dict  # catalog setting changes to apply on click
    fields: dict  # profile-field changes; sibling values are FILENAMES


def _rel_core(caps):
    t = {"ctx-size": Tier.RECOMMENDED, "n-gpu-layers": Tier.RECOMMENDED}
    for k in ("threads", "flash-attn", "temp", "port", "batch-size"):
        t[k] = Tier.TUNE
    return t


def _rel_moe(caps):
    if caps.is_moe:
        return {
            "n-cpu-moe": Tier.RECOMMENDED,
            "cpu-moe": Tier.TUNE,
            "override-tensor": Tier.TUNE,
            "n-cpu-ffn": Tier.TUNE,
        }
    return {
        "n-cpu-moe": Tier.NA,
        "cpu-moe": Tier.NA,
        "override-tensor": Tier.NA,
        "n-cpu-ffn": Tier.RECOMMENDED,
    }


def _rel_mtp(caps):
    if caps.has_mtp:
        t = {
            "spec-type": Tier.RECOMMENDED,
            "spec-draft-n-max": Tier.TUNE,
            "spec-draft-ngl": Tier.TUNE,
        }
        if caps.mtp_sibling:
            t["draft_model"] = Tier.RECOMMENDED
        return t
    return {
        k: Tier.NA
        for k in (
            "spec-type",
            "spec-draft-n-max",
            "spec-draft-ngl",
            "spec-draft-n-min",
            "cache-type-k-draft",
            "cache-type-v-draft",
        )
    }


def _rel_vision(caps):
    if caps.has_vision:
        return {"mmproj": Tier.RECOMMENDED, "no-mmproj-offload": Tier.TUNE}
    return {"mmproj": Tier.NA, "no-mmproj-offload": Tier.NA}


def _rel_swa(caps):
    if caps.has_swa:
        return {"swa-full": Tier.TUNE, "ctx-checkpoints": Tier.TUNE}
    return {"swa-full": Tier.NA}


_EMBED_NA_GROUPS = ("Sampling", "Speculative Decoding")


def _rel_embedding(caps):
    if not caps.is_embedding:
        return {}
    t = {
        "embeddings": Tier.RECOMMENDED,
        "pooling": Tier.RECOMMENDED,
        "ubatch-size": Tier.TUNE,
        "batch-size": Tier.TUNE,
        "mmproj": Tier.NA,
        "no-mmproj-offload": Tier.NA,
    }
    if caps.is_reranker:
        t["reranking"] = Tier.RECOMMENDED
    for key, s in CATALOG.items():
        if s.group in _EMBED_NA_GROUPS:
            t[key] = Tier.NA
    return t


# _rel_embedding MUST stay last: its N/A tiers for an embedding model
# override earlier contributors (e.g. vision's mmproj=RECOMMENDED).
RELEVANCE_CONTRIBUTORS = [
    _rel_core,
    _rel_moe,
    _rel_mtp,
    _rel_vision,
    _rel_swa,
    _rel_embedding,
]


def relevance(caps: ModelCaps) -> dict:
    t: dict = {}
    for contrib in RELEVANCE_CONTRIBUTORS:
        t.update(contrib(caps))
    return t


# Short, human reasons per setting group. Keyed by membership in the relevance
# sub-maps so the dot's hover can explain *why* a setting is (not) suggested.
_MOE_KEYS = ("n-cpu-moe", "cpu-moe", "override-tensor")
_FFN_KEYS = ("n-cpu-ffn",)
_MTP_KEYS = (
    "spec-type",
    "spec-draft-n-max",
    "spec-draft-ngl",
    "spec-draft-n-min",
    "cache-type-k-draft",
    "cache-type-v-draft",
    "draft_model",
)


def _reason_for(key: str, tier: "Tier", caps) -> str:
    if tier == Tier.NA:
        if key in _MOE_KEYS:
            return "Not applicable: not a MoE model."
        if key in _MTP_KEYS:
            return "Not applicable: model has no MTP/draft head."
        return "Not applicable to this model."
    if key in _FFN_KEYS:
        if tier == Tier.RECOMMENDED:
            return "Dense model: offload dense FFN layers to CPU to fit VRAM."
        return (
            "MoE model: prefer n-cpu-moe; dense FFN offload reaches the "
            "non-expert layers only."
        )
    if key in _MOE_KEYS:
        return "MoE model: offload experts to CPU to fit VRAM."
    if key in _MTP_KEYS:
        return "Model ships an MTP/draft head: enable speculative decoding."
    if tier == Tier.RECOMMENDED:
        return "Recommended for this model."
    return "Worth tuning for this model."


def describe_relevance(caps: ModelCaps) -> dict:
    """Like relevance(), but each key maps to (Tier, human reason)."""
    return {k: (t, _reason_for(k, t, caps)) for k, t in relevance(caps).items()}


def _sug_mtp(caps, settings, mmproj_set, draft_set):
    out = []
    if caps.has_mtp_infile and settings.get("spec-type") != "draft-mtp":
        out.append(
            Suggestion(
                "MTP head bundled: set spec-type = draft-mtp",
                {"spec-type": "draft-mtp"},
                {},
            )
        )
    if caps.mtp_sibling and not draft_set:
        out.append(
            Suggestion(
                f"MTP head file {caps.mtp_sibling} found: load as draft + set draft-mtp",
                {"spec-type": "draft-mtp"},
                {"draft_model": caps.mtp_sibling},
            )
        )
    return out


def _sug_vision(caps, settings, mmproj_set, draft_set):
    if caps.mmproj_sibling and not mmproj_set:
        return [
            Suggestion(
                f"Vision projector {caps.mmproj_sibling} found: set mmproj",
                {},
                {"mmproj": caps.mmproj_sibling},
            )
        ]
    return []


def _sug_ctx(caps, settings, mmproj_set, draft_set):
    ctx = settings.get("ctx-size") or 0
    if caps.ctx_train and ctx > caps.ctx_train:
        return [
            Suggestion(
                f"ctx-size exceeds trained max {caps.ctx_train}: cap it",
                {"ctx-size": caps.ctx_train},
                {},
            )
        ]
    return []


def _sug_embedding(caps, settings, mmproj_set, draft_set):
    if caps.is_reranker:
        have = (
            settings.get("reranking")
            and settings.get("embeddings")
            and settings.get("pooling") == "rank"
        )
        if have:
            return []
        return [
            Suggestion(
                "Reranker \u2192 enable rerank (--reranking + --pooling rank + --embeddings)",
                {"reranking": True, "pooling": "rank", "embeddings": True},
                {},
            )
        ]
    if caps.is_embedding and not settings.get("embeddings"):
        pool = caps.pooling_type or "mean"
        return [
            Suggestion(
                f"Embedding model \u2192 enable --embeddings, pooling = {pool}",
                {"embeddings": True, "pooling": pool},
                {},
            )
        ]
    return []


SUGGESTION_DETECTORS = [_sug_mtp, _sug_vision, _sug_ctx, _sug_embedding]


def suggestions(
    caps: ModelCaps, settings: dict, mmproj_set: bool = False, draft_set: bool = False
) -> list:
    out = []
    for det in SUGGESTION_DETECTORS:
        out.extend(det(caps, settings, mmproj_set, draft_set))
    return out
