from llama_launcher.services import benchmark as bm


def test_filler_prompt_deterministic_and_scales():
    a = bm.filler_prompt(128)
    assert a == bm.filler_prompt(128)          # deterministic
    assert len(bm.filler_prompt(512)) > len(a)  # bigger target → longer text
    assert a.strip()                            # non-empty


def _t(prompt_n, prompt_ms, predicted_n, predicted_ms):
    return {"prompt_n": prompt_n, "prompt_ms": prompt_ms,
            "predicted_n": predicted_n, "predicted_ms": predicted_ms}


def test_row_from_timings_averages_tok_s():
    # 512 tok in 1000ms -> 512 tok/s; 128 tok in 2000ms -> 64 tok/s
    s = _t(512, 1000, 128, 2000)
    row = bm.row_from_timings(512, [s, s, s])
    assert row.target_size == 512 and row.prompt_n == 512
    assert abs(row.pp_tok_s - 512.0) < 1e-6
    assert abs(row.gen_tok_s - 64.0) < 1e-6
    assert abs(row.total_s - 3.0) < 1e-6        # (1000+2000)/1000


def test_row_from_timings_guards_zero_ms():
    row = bm.row_from_timings(128, [_t(128, 0, 64, 0)])
    assert row.pp_tok_s == 0.0 and row.gen_tok_s == 0.0
