from dataclasses import dataclass

_BYTES_PER_ELEM = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q8_0": 1.0625, "q5_1": 0.75, "q5_0": 0.6875,
    "q4_1": 0.5625, "q4_0": 0.5625, "iq4_nl": 0.5625,
}


def bytes_per_elem(quant: str) -> float:
    return _BYTES_PER_ELEM.get((quant or "").lower(), 2.0)


@dataclass
class VramEstimate:
    kv_bytes: int
    weights_bytes: int
    overhead_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.kv_bytes + self.weights_bytes + self.overhead_bytes


def kv_cache_bytes(n_layers, n_head_kv, head_dim, ctx, k_quant="f16", v_quant="f16") -> int:
    per = int(n_layers) * int(ctx) * int(n_head_kv) * int(head_dim)
    return int(per * bytes_per_elem(k_quant) + per * bytes_per_elem(v_quant))


def estimate(*, n_layers, n_head, n_head_kv, n_embd, ctx,
             k_quant="f16", v_quant="f16", weights_bytes=0,
             overhead_bytes=536870912) -> VramEstimate:
    head_dim = (int(n_embd) // int(n_head)) if n_head else 0
    kv = kv_cache_bytes(n_layers, n_head_kv, head_dim, ctx, k_quant, v_quant)
    return VramEstimate(kv_bytes=kv, weights_bytes=int(weights_bytes),
                        overhead_bytes=int(overhead_bytes))


def fits(estimate_bytes: int, free_bytes: int) -> tuple[bool, int]:
    margin = int(free_bytes) - int(estimate_bytes)
    return (margin >= 0, margin)
