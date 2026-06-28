from dataclasses import dataclass
from enum import Enum

from .gguf import GgufMeta


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
    return ModelCaps(
        is_moe=bool(meta.expert_count and meta.expert_count > 0),
        expert_count=meta.expert_count,
        has_mtp_infile=bool(meta.nextn_predict_layers and meta.nextn_predict_layers > 0),
        mtp_sibling=_match(sibling_filenames, "mtp"),
        mmproj_sibling=_match(sibling_filenames, "mmproj"),
        has_swa=bool(meta.sliding_window and meta.sliding_window > 0),
        sliding_window=meta.sliding_window,
        ctx_train=meta.ctx_train,
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


def relevance(caps: ModelCaps) -> dict:
    t: dict = {"ctx-size": Tier.RECOMMENDED, "n-gpu-layers": Tier.RECOMMENDED}
    for k in ("threads", "flash-attn", "temp", "port", "batch-size"):
        t[k] = Tier.TUNE
    if caps.is_moe:
        t["n-cpu-moe"] = Tier.RECOMMENDED
        t["cpu-moe"] = t["override-tensor"] = Tier.TUNE
    else:
        for k in ("n-cpu-moe", "cpu-moe", "override-tensor"):
            t[k] = Tier.NA
    if caps.has_mtp:
        t["spec-type"] = Tier.RECOMMENDED
        t["spec-draft-n-max"] = t["spec-draft-ngl"] = Tier.TUNE
        if caps.mtp_sibling:
            t["draft_model"] = Tier.RECOMMENDED
    else:
        for k in ("spec-type", "spec-draft-n-max", "spec-draft-ngl",
                  "spec-draft-n-min", "cache-type-k-draft", "cache-type-v-draft"):
            t[k] = Tier.NA
    if caps.has_vision:
        t["mmproj"] = Tier.RECOMMENDED
        t["no-mmproj-offload"] = Tier.TUNE
    else:
        t["mmproj"] = t["no-mmproj-offload"] = Tier.NA
    if caps.has_swa:
        t["swa-full"] = t["ctx-checkpoints"] = Tier.TUNE
    else:
        t["swa-full"] = Tier.NA
    return t


def suggestions(caps: ModelCaps, settings: dict, mmproj_set: bool, draft_set: bool) -> list:
    out = []
    if caps.has_mtp_infile and settings.get("spec-type") != "draft-mtp":
        out.append(Suggestion("MTP head bundled — set spec-type = draft-mtp",
                              {"spec-type": "draft-mtp"}, {}))
    if caps.mtp_sibling and not draft_set:
        out.append(Suggestion(f"MTP head file {caps.mtp_sibling} found — load as draft + set draft-mtp",
                              {"spec-type": "draft-mtp"}, {"draft_model": caps.mtp_sibling}))
    if caps.mmproj_sibling and not mmproj_set:
        out.append(Suggestion(f"Vision projector {caps.mmproj_sibling} found — set mmproj",
                              {}, {"mmproj": caps.mmproj_sibling}))
    ctx = settings.get("ctx-size") or 0
    if caps.ctx_train and ctx > caps.ctx_train:
        out.append(Suggestion(f"ctx-size exceeds trained max {caps.ctx_train} — cap it",
                              {"ctx-size": caps.ctx_train}, {}))
    return out
