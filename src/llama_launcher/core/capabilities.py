from dataclasses import dataclass
from enum import Enum

from .gguf import GgufMeta


EMBEDDING_ARCHS = frozenset({
    "bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2", "jina-bert",
    "gte", "gte-qwen2", "roberta", "xlm-roberta", "distilbert", "mpnet",
})
_POOLING = {0: "none", 1: "mean", 2: "cls", 3: "last", 4: "rank"}


@dataclass(frozen=True)
class ModelCaps:
    is_moe: bool = False
    expert_count: int | None = None
    has_mtp_infile: bool = False
    mtp_sibling: str | None = None        # filename of a *mtp*.gguf sibling
    mmproj_sibling: str | None = None     # filename of an mmproj*.gguf sibling
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
        has_mtp_infile=bool(meta.nextn_predict_layers and meta.nextn_predict_layers > 0),
        mtp_sibling=_match(sibling_filenames, "mtp"),
        mmproj_sibling=_match(sibling_filenames, "mmproj"),
        has_swa=bool(meta.sliding_window and meta.sliding_window > 0),
        sliding_window=meta.sliding_window,
        ctx_train=meta.ctx_train,
        is_embedding=(meta.arch in EMBEDDING_ARCHS or pooling is not None),
        is_reranker=(pooling == "rank"),
        pooling_type=pooling,
    )


class Tier(str, Enum):
    RECOMMENDED = "recommended"   # set this for this model
    TUNE = "tune"                 # worth tuning
    NEUTRAL = "neutral"           # default styling (UI default; not returned)
    NA = "na"                     # not applicable for this model


@dataclass(frozen=True)
class Suggestion:
    text: str
    settings: dict     # catalog setting changes to apply on click
    fields: dict       # profile-field changes; sibling values are FILENAMES


def _rel_core(caps):
    t = {"ctx-size": Tier.RECOMMENDED, "n-gpu-layers": Tier.RECOMMENDED}
    for k in ("threads", "flash-attn", "temp", "port", "batch-size"):
        t[k] = Tier.TUNE
    return t


def _rel_moe(caps):
    if caps.is_moe:
        return {"n-cpu-moe": Tier.RECOMMENDED, "cpu-moe": Tier.TUNE,
                "override-tensor": Tier.TUNE}
    return {k: Tier.NA for k in ("n-cpu-moe", "cpu-moe", "override-tensor")}


def _rel_mtp(caps):
    if caps.has_mtp:
        t = {"spec-type": Tier.RECOMMENDED, "spec-draft-n-max": Tier.TUNE,
             "spec-draft-ngl": Tier.TUNE}
        if caps.mtp_sibling:
            t["draft_model"] = Tier.RECOMMENDED
        return t
    return {k: Tier.NA for k in ("spec-type", "spec-draft-n-max", "spec-draft-ngl",
                                 "spec-draft-n-min", "cache-type-k-draft",
                                 "cache-type-v-draft")}


def _rel_vision(caps):
    if caps.has_vision:
        return {"mmproj": Tier.RECOMMENDED, "no-mmproj-offload": Tier.TUNE}
    return {"mmproj": Tier.NA, "no-mmproj-offload": Tier.NA}


def _rel_swa(caps):
    if caps.has_swa:
        return {"swa-full": Tier.TUNE, "ctx-checkpoints": Tier.TUNE}
    return {"swa-full": Tier.NA}


RELEVANCE_CONTRIBUTORS = [_rel_core, _rel_moe, _rel_mtp, _rel_vision, _rel_swa]


def relevance(caps: ModelCaps) -> dict:
    t: dict = {}
    for contrib in RELEVANCE_CONTRIBUTORS:
        t.update(contrib(caps))
    return t


def _sug_mtp(caps, settings, mmproj_set, draft_set):
    out = []
    if caps.has_mtp_infile and settings.get("spec-type") != "draft-mtp":
        out.append(Suggestion("MTP head bundled — set spec-type = draft-mtp",
                              {"spec-type": "draft-mtp"}, {}))
    if caps.mtp_sibling and not draft_set:
        out.append(Suggestion(f"MTP head file {caps.mtp_sibling} found — load as draft + set draft-mtp",
                              {"spec-type": "draft-mtp"}, {"draft_model": caps.mtp_sibling}))
    return out


def _sug_vision(caps, settings, mmproj_set, draft_set):
    if caps.mmproj_sibling and not mmproj_set:
        return [Suggestion(f"Vision projector {caps.mmproj_sibling} found — set mmproj",
                          {}, {"mmproj": caps.mmproj_sibling})]
    return []


def _sug_ctx(caps, settings, mmproj_set, draft_set):
    ctx = settings.get("ctx-size") or 0
    if caps.ctx_train and ctx > caps.ctx_train:
        return [Suggestion(f"ctx-size exceeds trained max {caps.ctx_train} — cap it",
                          {"ctx-size": caps.ctx_train}, {})]
    return []


SUGGESTION_DETECTORS = [_sug_mtp, _sug_vision, _sug_ctx]


def suggestions(caps: ModelCaps, settings: dict,
                mmproj_set: bool = False, draft_set: bool = False) -> list:
    out = []
    for det in SUGGESTION_DETECTORS:
        out.extend(det(caps, settings, mmproj_set, draft_set))
    return out
