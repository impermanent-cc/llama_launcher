import pytest

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


class _FakeClient:
    def __init__(self, timings, fail_on=None):
        self.timings, self.fail_on, self.calls = timings, fail_on, 0

    def __call__(self, prompt, n_predict):
        self.calls += 1
        if self.fail_on and self.calls == self.fail_on:
            raise RuntimeError("boom")
        return {"timings": self.timings}


def test_run_benchmark_warmup_discarded_and_rows_per_size():
    c = _FakeClient(_t(512, 1000, 128, 2000))
    run = bm.run_benchmark(c, sizes=[128, 512], n_predict=128, warmup=1, repeats=2,
                           snapshot={"model": "m"}, timestamp="2026-08-07T00:00:00")
    assert [r.target_size for r in run.rows] == [128, 512]
    assert c.calls == 2 * (1 + 2)               # 2 sizes × (warmup + repeats)
    assert run.snapshot == {"model": "m"} and run.n_predict == 128


def test_run_benchmark_client_failure_aborts():
    c = _FakeClient(_t(512, 1000, 128, 2000), fail_on=1)
    with pytest.raises(bm.BenchmarkError):
        bm.run_benchmark(c, sizes=[128], n_predict=8, warmup=0, repeats=1,
                         snapshot={}, timestamp="t")


def test_run_benchmark_cancel_between_calls():
    c = _FakeClient(_t(512, 1000, 128, 2000))
    with pytest.raises(bm.BenchmarkError):
        bm.run_benchmark(c, sizes=[128, 512], n_predict=8, warmup=0, repeats=1,
                         snapshot={}, timestamp="t", should_cancel=lambda: True)
    assert c.calls == 0                          # cancelled before the first call


def test_build_snapshot_server_reads_profile_settings():
    from llama_launcher.core.spec import Profile, Runtime
    p = Profile(name="s", image="img:tag", runtime=Runtime(), mode="server",
                model="/models/qwen.gguf", settings={"n-gpu-layers": "99", "flash-attn": "on"})
    snap = bm.build_snapshot(p)
    assert snap["model"] == "qwen.gguf" and snap["image"] == "img:tag"
    assert snap["ngl"] == "99" and snap["fa"] == "on" and snap["ctx"] is None
