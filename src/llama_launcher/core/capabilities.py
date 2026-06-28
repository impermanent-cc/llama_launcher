from dataclasses import dataclass

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
