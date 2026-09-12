import subprocess

from llama_launcher.core.spec import Mount, Profile, Runtime
from llama_launcher.core.sweep import Sweep
from llama_launcher.services import sweep as svc
from llama_launcher.services.benchmark import BenchmarkRow, BenchmarkRun
from llama_launcher.services.headless import LaunchResult

LOG_OK = "load_tensors:        CUDA0 model buffer size =   100.00 MiB\nmain: server is listening\n"


def _profile():
    return Profile(
        name="p",
        image="img",
        runtime=Runtime(binary="podman"),
        mounts=[Mount(host="/h", container="/models", role="model")],
        model="/models/m.gguf",
        settings={"port": 8080, "ctx-size": 4096},
    )


def _probes(*, ready=(True, True, True), launch_ok=True):
    calls = {"launch": [], "stop": [], "bench": [], "clock": [0.0]}
    ready_iter = iter(ready)

    def launch(profile, base_dir, binary):
        calls["launch"].append((profile.name, dict(profile.settings)))
        return LaunchResult(
            launch_ok,
            "llama-p-sweep",
            "127.0.0.1",
            8080,
            [],
            None if launch_ok else "boom",
        )

    def wait_ready(host, port, timeout, should_cancel):
        return next(ready_iter, True)

    def read_log(name, binary):
        return LOG_OK

    def run_benchmark(
        client,
        sizes,
        n_predict,
        warmup,
        repeats,
        snapshot,
        timestamp,
        should_cancel=None,
    ):
        calls["bench"].append(snapshot)
        return BenchmarkRun(
            timestamp,
            list(sizes),
            n_predict,
            warmup,
            repeats,
            [BenchmarkRow(s, s, 500.0, 20.0 + len(calls["bench"]), 1.0) for s in sizes],
            snapshot,
        )

    def stop(name, binary, timeout):
        calls["stop"].append(name)

    def clock():
        calls["clock"][0] += 5.0
        return calls["clock"][0]

    return calls, dict(
        launch=launch,
        wait_ready=wait_ready,
        read_log=read_log,
        run_benchmark=run_benchmark,
        stop=stop,
        clock=clock,
    )


def test_sweep_profile_is_a_renamed_copy_with_the_count():
    p = _profile()
    q = svc.sweep_profile(p, 6, "n-cpu-ffn")
    assert q.name == "p sweep" and q.settings["n-cpu-ffn"] == 6
    assert p.settings.get("n-cpu-ffn") is None and q is not p


def test_sweep_profile_raises_verbosity_to_print_buffer_sizes():
    """A profile with no verbosity setting gets the level that makes
    llama-server print its load-time buffer-size lines."""
    q = svc.sweep_profile(_profile(), 6, "n-cpu-ffn")
    assert q.settings["verbosity"] == 4


def test_sweep_profile_keeps_a_higher_verbosity():
    p = _profile()
    p.settings["verbosity"] = 6
    q = svc.sweep_profile(p, 6, "n-cpu-ffn")
    assert q.settings["verbosity"] == 6


def test_sweep_profile_leaves_verbosity_unset_for_ik_llama_cpp():
    """ik_llama.cpp prints its load-time buffer lines without the verbosity
    flag, so a sweep never sets it for that engine."""
    p = _profile()
    p.runtime = Runtime(binary="podman", engine="ik_llama.cpp")
    q = svc.sweep_profile(p, 6, "n-cpu-ffn")
    assert q.settings.get("verbosity") is None


def test_sweep_profile_clears_log_settings_that_hide_the_buffer_lines():
    """--log-disable prints nothing and --log-colors on wraps every line in
    escapes, either of which leaves the sweep with no measurement."""
    p = _profile()
    p.settings["log-disable"] = True
    p.settings["log-colors"] = "on"
    q = svc.sweep_profile(p, 6, "n-cpu-ffn")
    assert "log-disable" not in q.settings
    assert "log-colors" not in q.settings
    assert p.settings["log-disable"] is True


def test_sweep_profile_negates_a_raw_log_jsonl():
    """A raw --log-jsonl gets --no-log-jsonl appended on the copy, leaving
    the caller's raw_args untouched."""
    p = _profile()
    p.raw_args = "--log-jsonl"
    q = svc.sweep_profile(p, 6, "n-cpu-ffn")
    assert q.raw_args.split() == ["--log-jsonl", "--no-log-jsonl"]
    assert p.raw_args == "--log-jsonl"


def test_sweep_profile_adds_no_negation_without_log_jsonl():
    """--no-log-jsonl exists only from b10902, so a build that never saw
    --log-jsonl would reject it and fail the whole sweep."""
    p = _profile()
    p.raw_args = "--foo bar"
    q = svc.sweep_profile(p, 6, "n-cpu-ffn")
    assert "--no-log-jsonl" not in q.raw_args


def test_sweep_profile_adds_no_negation_when_already_present():
    """A profile that already carries both flags gets nothing appended,
    since the later token already wins."""
    p = _profile()
    p.raw_args = "--log-jsonl --no-log-jsonl"
    q = svc.sweep_profile(p, 6, "n-cpu-ffn")
    assert q.raw_args.split() == ["--log-jsonl", "--no-log-jsonl"]


def test_sweep_profile_is_idempotent_on_its_own_output():
    """Applying sweep_profile to its own result adds no further negation,
    so chaining it never accumulates --no-log-jsonl tokens."""
    p = _profile()
    p.raw_args = "--log-jsonl"
    q1 = svc.sweep_profile(p, 6, "n-cpu-ffn")
    q2 = svc.sweep_profile(q1, 6, "n-cpu-ffn")
    assert q2.raw_args.split() == ["--log-jsonl", "--no-log-jsonl"]


def test_sweep_profile_leaves_log_file_alone():
    """--log-file still writes every line to stderr, so it does not hide the
    buffer lines and the profile's choice stands."""
    p = _profile()
    p.settings["log-file"] = "/tmp/server.log"
    q = svc.sweep_profile(p, 6, "n-cpu-ffn")
    assert q.settings["log-file"] == "/tmp/server.log"


def test_run_sweep_records_every_point_and_stops_each_container():
    calls, probes = _probes()
    seen = []
    sweep = svc.run_sweep(
        _profile(),
        "/base",
        knob="n-cpu-ffn",
        counts=[0, 2, 4],
        bench_cfg={"sizes": [128, 512], "n_predict": 8, "warmup": 0, "repeats": 1},
        estimate_for=lambda count: ((1000 + count,), 50),
        ready_timeout=30,
        on_point=seen.append,
        **probes,
    )
    assert isinstance(sweep, Sweep) and [p.count for p in sweep.points] == [0, 2, 4]
    assert all(p.status == "ok" for p in sweep.points)
    assert sweep.points[1].measured.cards[0]["model"] == 100 * 1024 * 1024
    assert (
        sweep.points[1].estimated_cards == (1002,)
        and sweep.points[1].estimated_ram == 50
    )
    assert sweep.points[0].ready_seconds == 5.0
    assert [s for s in calls["stop"]] == ["llama-p-sweep"] * 3
    assert [c[1]["n-cpu-ffn"] for c in calls["launch"]] == [0, 2, 4]
    assert [c[1]["verbosity"] for c in calls["launch"]] == [4, 4, 4]
    assert [p.status for p in seen] == ["running", "ok"] * 3
    assert [p.count for p in seen if p.status == "ok"] == [0, 2, 4]


def test_run_sweep_records_a_failed_point_and_continues():
    calls, probes = _probes(ready=(False, True))
    sweep = svc.run_sweep(
        _profile(),
        "/base",
        knob="n-cpu-ffn",
        counts=[0, 2],
        bench_cfg={"sizes": [128], "n_predict": 8, "warmup": 0, "repeats": 1},
        estimate_for=lambda c: ((0,), 0),
        ready_timeout=1,
        **probes,
    )
    assert [p.status for p in sweep.points] == ["failed", "ok"]
    assert sweep.points[0].error == "main: server is listening"
    assert sweep.points[0].ready_seconds is None and sweep.points[0].rows == ()
    assert len(calls["stop"]) == 2


def test_run_sweep_launch_failure_is_a_failed_point():
    calls, probes = _probes(launch_ok=False)
    sweep = svc.run_sweep(
        _profile(),
        "/base",
        knob="n-cpu-moe",
        counts=[3],
        bench_cfg={"sizes": [128], "n_predict": 8, "warmup": 0, "repeats": 1},
        estimate_for=lambda c: ((0,), 0),
        ready_timeout=1,
        **probes,
    )
    assert sweep.points[0].status == "failed" and sweep.points[0].error == "boom"
    # The name may belong to something else, so a refused launch removes nothing.
    assert calls["stop"] == []


def test_run_sweep_cancel_records_the_point_as_cancelled():
    """The point in flight when a cancel lands is cancelled, not failed, and
    its container is still stopped."""
    calls, probes = _probes()
    flags = iter([False, True])
    seen = []
    sweep = svc.run_sweep(
        _profile(),
        "/base",
        knob="n-cpu-ffn",
        counts=[0, 2, 4],
        bench_cfg={"sizes": [128], "n_predict": 8, "warmup": 0, "repeats": 1},
        estimate_for=lambda c: ((0,), 0),
        ready_timeout=1,
        should_cancel=lambda: next(flags, True),
        on_point=seen.append,
        **probes,
    )
    assert [p.count for p in sweep.points] == [0]
    assert sweep.points[0].status == "cancelled"
    assert sweep.points[0].error == "cancelled"
    assert [p.status for p in seen] == ["running", "cancelled"]
    assert calls["stop"] == ["llama-p-sweep"]


def test_wait_ready_gives_up_as_soon_as_the_container_is_not_running(monkeypatch):
    """A server that dies while loading costs one poll, not the whole ready
    timeout."""
    monkeypatch.setattr(svc.headless, "probe_health", lambda port, host="": "starting")
    probed = []
    monkeypatch.setattr(
        svc.runtime,
        "container_state",
        lambda name, binary: probed.append((name, binary)) or "stopped",
    )
    slept = []
    monkeypatch.setattr(svc.headless.time, "sleep", slept.append)

    ready = svc._wait_ready(
        "127.0.0.1", 8080, 600, None, name="llama-p-sweep", binary="podman"
    )

    assert ready is False
    assert probed == [("llama-p-sweep", "podman")]
    assert slept == []


def test_wait_ready_returns_true_on_a_ready_server(monkeypatch):
    monkeypatch.setattr(svc.headless, "probe_health", lambda port, host="": "ready")
    monkeypatch.setattr(svc.runtime, "container_state", lambda name, binary: "running")
    assert (
        svc._wait_ready("127.0.0.1", 8080, 600, None, name="c", binary="podman") is True
    )


def test_run_sweep_benchmark_error_is_a_failed_point():
    from llama_launcher.services.benchmark import BenchmarkError

    calls, probes = _probes()

    def bad(*a, **k):
        raise BenchmarkError("timeout")

    probes["run_benchmark"] = bad
    sweep = svc.run_sweep(
        _profile(),
        "/base",
        knob="n-cpu-ffn",
        counts=[0],
        bench_cfg={"sizes": [128], "n_predict": 8, "warmup": 0, "repeats": 1},
        estimate_for=lambda c: ((0,), 0),
        ready_timeout=1,
        **probes,
    )
    assert sweep.points[0].status == "failed" and "timeout" in sweep.points[0].error
    assert calls["stop"] == ["llama-p-sweep"]


def test_run_sweep_unexpected_error_still_stops_the_container():
    calls, probes = _probes()

    def bad(name, binary):
        raise RuntimeError("boom")

    probes["read_log"] = bad
    sweep = svc.run_sweep(
        _profile(),
        "/base",
        knob="n-cpu-ffn",
        counts=[0],
        bench_cfg={"sizes": [128], "n_predict": 8, "warmup": 0, "repeats": 1},
        estimate_for=lambda c: ((0,), 0),
        ready_timeout=1,
        **probes,
    )
    assert sweep.points[0].status == "failed"
    assert sweep.points[0].error == "RuntimeError: boom"
    assert calls["stop"] == ["llama-p-sweep"]


def test_read_log_joins_stdout_and_stderr_and_swallows_failures(monkeypatch):
    seen = {}

    def run(argv, **k):
        seen["argv"] = argv
        seen["kw"] = k
        return subprocess.CompletedProcess(argv, 0, "out\n", "err\n")

    monkeypatch.setattr(svc.subprocess, "run", run)
    assert svc._read_log("llama-x-sweep", "podman") == "out\nerr\n"
    assert seen["argv"][-2:] == ["logs", "llama-x-sweep"]
    assert seen["kw"]["timeout"] == 30

    monkeypatch.setattr(
        svc.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()),
    )
    assert svc._read_log("llama-x-sweep", "podman") == ""


def test_stop_and_remove_runs_stop_then_rm_and_survives_a_timeout(monkeypatch):
    calls = []

    def run(argv, **k):
        calls.append((argv[1], k["timeout"]))
        if argv[1] == "stop":
            raise subprocess.TimeoutExpired(argv, k["timeout"])
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(svc.subprocess, "run", run)
    svc._stop_and_remove("llama-x-sweep", "podman", 7)
    assert [c[0] for c in calls] == ["stop", "rm"]
    assert all(c[1] == 37 for c in calls)
