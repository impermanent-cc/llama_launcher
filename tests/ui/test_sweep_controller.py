"""The Benchmark controller's offload sweep: the refusals, the prefill from
the memory estimate, the synchronous run seam, the estimate closure and
Apply."""

import threading
import time
from types import SimpleNamespace

import pytest

from llama_launcher.core import memory_fit, vram
from llama_launcher.core import sweep as core_sweep
from llama_launcher.core.gguf import GgufMeta, TensorInfo
from llama_launcher.core.spec import Mount, Profile, RouterMember, Runtime
from llama_launcher.services import benchmark_store, runtime, sweep_store
from llama_launcher.services import sweep as sweep_service
from llama_launcher.services.gpu import GpuStat
from llama_launcher.store.profiles import default_base_dir
from llama_launcher.ui.controllers import benchmark_controller
from llama_launcher.ui.main_window import MainWindow

MIB = 1024 * 1024
GIB = 1024 * MIB
CFG = {"start": 0, "stop": 4, "step": 2, "ready_timeout": 600}


@pytest.fixture
def win(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _meta(n_layers=8, moe=False):
    ts = [TensorInfo("token_embd.weight", 1, 0, 10 * MIB)]
    for i in range(n_layers):
        ts.append(TensorInfo(f"blk.{i}.attn_q.weight", 1, 0, 100 * MIB))
        name = f"blk.{i}.ffn_up_exps.weight" if moe else f"blk.{i}.ffn_up.weight"
        ts.append(TensorInfo(name, 1, 0, 400 * MIB))
    ts.append(TensorInfo("output.weight", 1, 0, 100 * MIB))
    return GgufMeta(
        arch="qwen3moe" if moe else "llama",
        n_layers=n_layers,
        n_head=8,
        n_head_kv=8,
        n_embd=512,
        ctx_train=32768,
        n_ff=256,
        n_vocab=1000,
        expert_count=64 if moe else None,
        tensors=tuple(ts),
    )


def _gpu(free_mib=8192):
    return GpuStat(
        name="card",
        mem_used_mib=0,
        mem_total_mib=16384,
        mem_free_mib=free_mib,
        util_pct=0,
        temp_c=0,
    )


def _server_profile(name="Solo", **runtime_kwargs):
    return Profile(
        name=name,
        image="img",
        model="/models/a.gguf",
        mounts=[Mount(host="/h", container="/models")],
        runtime=Runtime(**runtime_kwargs),
        settings={"port": 8080},
    )


def _prepared(win, moe=False, name="Solo"):
    """A loaded single-server profile with a model estimate the sweep can
    read: metadata, weight size and one visible card."""
    p = _server_profile(name)
    win._configure_panel.load_profile(p)
    win._configure_panel._fit_meta = _meta(moe=moe)
    win._configure_panel._fit_weights = 4 * GIB
    win._configure_panel._fit_gpus = [_gpu()]
    win._configure_panel._fit_ram = 64 * GIB
    return p


def _point(count, gen):
    return core_sweep.SweepPoint(
        count,
        "ok",
        5.0,
        (
            {
                "target_size": 128,
                "prompt_n": 128,
                "pp_tok_s": 800.0,
                "gen_tok_s": gen,
                "total_s": 1.0,
            },
        ),
        None,
        (12 * GIB,),
        2 * GIB,
        "",
    )


def _two_point_sweep(name="Solo", knob="n-cpu-ffn"):
    return core_sweep.Sweep(name, knob, "t", (_point(0, 30.0), _point(4, 40.0)), {})


def _stub_run_sweep(captured, emit_points=False):
    def fake(profile, base_dir, **kwargs):
        captured.update(kwargs)
        captured["profile"] = profile
        captured["base_dir"] = base_dir
        sweep = _two_point_sweep(profile.name, kwargs["knob"])
        if emit_points and kwargs.get("on_point") is not None:
            for pt in sweep.points:
                kwargs["on_point"](
                    core_sweep.SweepPoint(
                        pt.count,
                        "running",
                        None,
                        (),
                        None,
                        pt.estimated_cards,
                        pt.estimated_ram,
                        "",
                    )
                )
                kwargs["on_point"](pt)
        return sweep

    return fake


def test_sweep_refused_for_native_router_rpc_and_remote(win):
    router = Profile(
        name="Host",
        mode="router",
        image="img",
        members=[RouterMember(profile="Qwen")],
        settings={"port": 8080},
    )
    native = _server_profile(launch_mode="native", native_binary="/usr/bin/llama")
    rpc = _server_profile(launch_mode="rpc")
    remote = _server_profile(node="box2")
    for p in (router, native, rpc, remote):
        assert win._benchmark._sweep_availability(p) != ""

    win._configure_panel.load_profile(native)

    assert "container" in win.benchmark_panel.sweep_status.text()
    assert not win.benchmark_panel.sweep_run_btn.isEnabled()


def test_sweep_refused_while_instance_running(win, monkeypatch):
    p = _prepared(win)
    name = win._container_name()
    monkeypatch.setattr(
        runtime,
        "container_state",
        lambda n, b, connection="": "running" if n == name else "absent",
    )
    captured = {}

    win._benchmark._run_sweep_sync(CFG, run_sweep=_stub_run_sweep(captured))

    assert captured == {}  # the start was refused, not run
    assert sweep_store.load(default_base_dir(), p.name) is None
    assert "same port" in win.benchmark_panel.sweep_status.text()
    assert not win.benchmark_panel.sweep_run_btn.isEnabled()


def test_fit_render_never_probes_the_container_state(win, monkeypatch):
    """The refusal reason a render derives is the cheap one: only a sweep
    start may pay for a container-state subprocess."""
    probes = []
    monkeypatch.setattr(
        runtime,
        "container_state",
        lambda n, b, connection="": probes.append(n) or "absent",
    )

    win._configure_panel.load_profile(_server_profile())
    win._configure_panel._set_fit_line("")
    win._benchmark.refresh_sweep()

    assert probes == []


def test_sweep_refused_without_prompt_sizes(win):
    p = _prepared(win)
    win.benchmark_panel.bench_sizes.setText("  ")
    captured = {}

    win._benchmark._run_sweep_sync(CFG, run_sweep=_stub_run_sweep(captured))

    assert captured == {}
    assert sweep_store.load(default_base_dir(), p.name) is None
    assert "prompt size" in win.benchmark_panel.sweep_status.text()


def test_sweep_prefill_from_estimate(win, monkeypatch):
    _prepared(win)
    panel = win._configure_panel
    monkeypatch.setattr(
        memory_fit, "smallest_fitting_offload", lambda *a, **k: ("n-cpu-ffn", 12)
    )
    monkeypatch.setattr(
        panel, "_current_fit_report", lambda: SimpleNamespace(fits=False)
    )

    win._benchmark.sweep_prefill()

    cfg = win.benchmark_panel.sweep_config()
    assert (cfg["start"], cfg["stop"], cfg["step"]) == (12, 20, 2)
    assert "n-cpu-ffn" in win.benchmark_panel.sweep_knob_label.text()

    monkeypatch.setattr(
        panel, "_current_fit_report", lambda: SimpleNamespace(fits=True)
    )
    win._benchmark.sweep_prefill()
    cfg = win.benchmark_panel.sweep_config()
    assert (cfg["start"], cfg["stop"], cfg["step"]) == (0, 8, 2)

    panel._fit_meta = _meta(moe=True)
    monkeypatch.setattr(
        memory_fit, "smallest_fitting_offload", lambda *a, **k: ("n-cpu-moe", 3)
    )
    monkeypatch.setattr(
        panel, "_current_fit_report", lambda: SimpleNamespace(fits=False)
    )
    win._benchmark.sweep_prefill()
    cfg = win.benchmark_panel.sweep_config()
    assert (cfg["start"], cfg["stop"], cfg["step"]) == (3, 11, 2)
    assert "n-cpu-moe" in win.benchmark_panel.sweep_knob_label.text()


def test_run_sweep_sync_saves_and_shows(win):
    p = _prepared(win)
    captured = {}

    win._benchmark._run_sweep_sync(
        CFG, run_sweep=_stub_run_sweep(captured, emit_points=True)
    )

    stored = sweep_store.load(default_base_dir(), p.name)
    assert stored is not None
    assert [pt["count"] for pt in stored["points"]] == [0, 4]
    table = win.benchmark_panel.sweep_table
    assert table.rowCount() == 2
    assert "(best)" in table.item(1, 0).text()
    assert benchmark_store.load(default_base_dir(), p.name) == []
    assert captured["knob"] == "n-cpu-ffn"
    assert captured["counts"] == [0, 2, 4]


def test_apply_writes_count_and_saves(win, monkeypatch):
    _prepared(win)
    win._benchmark._run_sweep_sync(CFG, run_sweep=_stub_run_sweep({}))
    saves = []
    monkeypatch.setattr(win, "save_current_profile", lambda: saves.append(1))

    win._benchmark._on_sweep_apply(4)

    assert win._configure_panel._widgets["n-cpu-ffn"].value() == 4
    assert saves == [1]


def test_live_points_use_the_planned_card_count(win):
    """The header is rebuilt for the planned cards before the first point, so
    a live row never lands under the previous sweep's columns."""
    _prepared(win)
    win.benchmark_panel.show_sweep({"points": []}, 3)  # a stale 3-card header
    columns = []

    def fake(profile, base_dir, **kwargs):
        sweep = _two_point_sweep(profile.name, kwargs["knob"])
        kwargs["on_point"](sweep.points[0])
        columns.append(win.benchmark_panel.sweep_table.columnCount())
        return sweep

    win._benchmark._run_sweep_sync(CFG, run_sweep=fake)

    assert columns == [7]  # 5 base columns + one card + RAM


def test_apply_refuses_when_another_profile_is_loaded(win, monkeypatch):
    _prepared(win)
    win._benchmark._run_sweep_sync(CFG, run_sweep=_stub_run_sweep({}))
    win._configure_panel.name_edit.setText("Other")
    saves = []
    monkeypatch.setattr(win, "save_current_profile", lambda: saves.append(1))

    win._benchmark._on_sweep_apply(4)

    assert saves == []
    assert win._configure_panel._widgets["n-cpu-ffn"].value() == 0
    assert "Solo" in win.benchmark_panel.sweep_status.text()


def test_drain_removes_a_terminated_sweep_container(win, monkeypatch):
    """A worker that outlives the drain wait is terminated, so run_sweep's own
    cleanup never runs: the teardown removes the sweep container itself, with
    one detached `rm -f` it never waits for."""
    _prepared(win)
    started = threading.Event()

    def blocking(profile, base_dir, **kwargs):
        started.set()
        while True:
            time.sleep(0.05)

    monkeypatch.setattr(sweep_service, "run_sweep", blocking)
    argvs = []
    monkeypatch.setattr(
        benchmark_controller.subprocess,
        "Popen",
        lambda argv, **kw: argvs.append(argv),
    )

    win._benchmark._on_sweep_run(CFG)
    assert started.wait(5)
    win._benchmark.drain()

    # One command for the sweep container, and it is the detached rm -f.
    assert [a for a in argvs if a[-1] == "llama-solo-sweep"] == [
        ["podman", "rm", "-f", "llama-solo-sweep"]
    ]


def test_estimate_for_uses_the_panel_report(win):
    _prepared(win)
    captured = {}

    win._benchmark._run_sweep_sync(CFG, run_sweep=_stub_run_sweep(captured))

    estimate_for = captured["estimate_for"]
    cards0, ram0 = estimate_for(0)
    cards8, ram8 = estimate_for(8)
    assert len(cards0) == 1 and len(cards8) == 1
    assert cards8[0] < cards0[0]
    assert ram8 > ram0

    fit = win._configure_panel._fit_report_kwargs()
    est = vram.estimate_memory(
        fit["meta"],
        fit["weights_bytes"],
        settings={**fit["settings"], "n-cpu-ffn": 0},
        **{k: fit[k] for k in benchmark_controller._ESTIMATE_KEYS},
    )
    card = est.cards[0]
    # The card figure names the same buffers the load log measures: weights,
    # KV and compute, without the fixed per-card overhead.
    assert cards0[0] == card.weights + card.kv + card.compute
    assert cards0[0] == card.total - vram.CARD_OVERHEAD_BYTES
    assert ram0 == est.ram.total


def test_benchmark_refused_while_a_sweep_runs(win):
    """A benchmark started during a sweep never runs: the sweep container
    holds the profile's port."""
    _prepared(win)
    win._benchmark._sweep_thread = object()
    calls = []
    try:
        win._benchmark._run_benchmark_sync(
            {"sizes": [128], "n_predict": 8, "warmup": 0, "repeats": 1},
            run_benchmark=lambda *a, **k: calls.append(a),
        )
    finally:
        win._benchmark._sweep_thread = None

    assert calls == []
    assert win.benchmark_panel.bench_progress.text() == "A sweep is running."


def test_sweep_refused_when_raw_args_carry_the_knob(win, monkeypatch):
    """Raw arguments spelling the swept flag (either knob, alias included)
    refuse the start: the raw value would override every swept count."""
    monkeypatch.setattr(runtime, "container_state", lambda *a, **k: "absent")

    for moe, raw in ((False, "--n-cpu-ffn 12"), (True, "--n-cpu-moe 12")):
        p = _prepared(win, moe=moe)
        p.raw_args = raw
        reason = win._benchmark._sweep_start_refusal(p)
        assert "raw arguments" in reason
        assert raw.split()[0] in reason
        p.raw_args = ""
        assert win._benchmark._sweep_start_refusal(p) == ""

    p = _prepared(win, moe=True)
    p.raw_args = "-ncmoe 12"
    assert "raw arguments" in win._benchmark._sweep_start_refusal(p)


def test_sweep_with_the_knob_in_raw_args_never_runs(win, monkeypatch):
    monkeypatch.setattr(runtime, "container_state", lambda *a, **k: "absent")
    p = _prepared(win)
    p.raw_args = "--n-cpu-ffn 12"
    win._configure_panel.raw_edit.setText(p.raw_args)
    captured = {}

    win._benchmark._run_sweep_sync(CFG, run_sweep=_stub_run_sweep(captured))

    assert captured == {}
    assert sweep_store.load(default_base_dir(), p.name) is None
    assert "raw arguments" in win.benchmark_panel.sweep_status.text()


def test_refresh_keeps_a_user_entered_range(win):
    """A range typed into the panel survives refreshes whose estimate has not
    changed, so a debounced fit render cannot clobber it."""
    _prepared(win)
    win._benchmark.refresh_sweep()
    panel = win.benchmark_panel
    panel.sweep_from.setValue(5)
    panel.sweep_to.setValue(9)
    panel.sweep_step.setValue(1)

    win._benchmark.refresh_sweep()
    win._benchmark.refresh_sweep()

    cfg = panel.sweep_config()
    assert (cfg["start"], cfg["stop"], cfg["step"]) == (5, 9, 1)


def test_refresh_rewrites_the_range_when_the_estimate_changes(win, monkeypatch):
    _prepared(win)
    monkeypatch.setattr(
        win._configure_panel, "_current_fit_report", lambda: SimpleNamespace(fits=False)
    )
    monkeypatch.setattr(
        memory_fit, "smallest_fitting_offload", lambda *a, **k: ("n-cpu-ffn", 4)
    )

    win._benchmark.refresh_sweep()

    cfg = win.benchmark_panel.sweep_config()
    assert (cfg["start"], cfg["stop"], cfg["step"]) == (4, 12, 2)

    monkeypatch.setattr(
        memory_fit, "smallest_fitting_offload", lambda *a, **k: ("n-cpu-ffn", 6)
    )
    win._benchmark.refresh_sweep()

    cfg = win.benchmark_panel.sweep_config()
    assert (cfg["start"], cfg["stop"], cfg["step"]) == (6, 14, 2)


def test_refresh_keeps_a_failed_sweep_status(win):
    _prepared(win)
    win._benchmark.refresh_sweep()

    win._benchmark._on_sweep_failed("boom")
    win._benchmark.refresh_sweep()

    assert "Sweep failed: boom" in win.benchmark_panel.sweep_status.text()


def test_refresh_writes_nothing_while_a_sweep_runs(win, monkeypatch):
    """A refresh during a sweep touches neither the range nor the status: the
    range is the running sweep's and the status is its progress."""
    _prepared(win)
    win._benchmark.refresh_sweep()
    panel = win.benchmark_panel
    panel.sweep_from.setValue(5)
    panel.sweep_status.setText("Point 2 of 5")
    monkeypatch.setattr(
        win._configure_panel, "_current_fit_report", lambda: SimpleNamespace(fits=False)
    )
    monkeypatch.setattr(
        memory_fit, "smallest_fitting_offload", lambda *a, **k: ("n-cpu-ffn", 7)
    )
    win._benchmark._sweep_thread = object()
    try:
        win._benchmark.refresh_sweep()
    finally:
        win._benchmark._sweep_thread = None

    assert panel.sweep_config()["start"] == 5
    assert panel.sweep_status.text() == "Point 2 of 5"


def test_sweep_refused_when_the_engine_lacks_the_knob(win):
    """A dense model on ik_llama.cpp has no --n-cpu-ffn to vary; a MoE model,
    or the same profile on llama.cpp, sweeps."""
    dense = _prepared(win)
    dense.runtime.engine = "ik_llama.cpp"

    assert win._benchmark._sweep_availability(dense) == (
        "ik_llama.cpp has no --n-cpu-ffn: sweep a MoE model or use llama.cpp."
    )

    dense.runtime.engine = "llama.cpp"
    assert win._benchmark._sweep_availability(dense) == ""

    moe = _prepared(win, moe=True)
    moe.runtime.engine = "ik_llama.cpp"
    assert win._benchmark._sweep_availability(moe) == ""


def test_sweep_refused_while_a_previous_sweep_container_runs(win, monkeypatch):
    """A sweep container left running from an earlier sweep holds the port and
    the name, so the start refuses by name."""
    p = _prepared(win)
    monkeypatch.setattr(
        runtime,
        "container_state",
        lambda n, b, connection="": "running" if n == "llama-solo-sweep" else "absent",
    )

    reason = win._benchmark._sweep_start_refusal(p)

    assert reason == (
        "A previous sweep container is still running: stop llama-solo-sweep first."
    )


def test_a_cancelled_sweep_is_never_stored(win, monkeypatch):
    """A cancelled sweep stops wherever the cancel landed, so it is shown but
    never written over the profile's stored sweep."""
    monkeypatch.setattr(runtime, "container_state", lambda *a, **k: "absent")
    p = _prepared(win)

    def fake(profile, base_dir, **kwargs):
        return core_sweep.Sweep(
            profile.name,
            kwargs["knob"],
            "t",
            (_point(0, 30.0), _cancelled_point(2)),
            {},
        )

    win._benchmark._run_sweep_sync(CFG, run_sweep=fake)

    assert sweep_store.load(default_base_dir(), p.name) is None
    assert win.benchmark_panel.sweep_table.rowCount() == 2
    assert win.benchmark_panel.sweep_status.text() == "Sweep cancelled."


def test_a_cancelled_sweep_leaves_the_stored_one_alone(win, monkeypatch):
    monkeypatch.setattr(runtime, "container_state", lambda *a, **k: "absent")
    p = _prepared(win)
    base = default_base_dir()
    sweep_store.save(base, p.name, _two_point_sweep(p.name))
    before = sweep_store.sweep_path(base, p.name).read_text()

    def fake(profile, base_dir, **kwargs):
        return core_sweep.Sweep(
            profile.name, kwargs["knob"], "t2", (_cancelled_point(0),), {}
        )

    win._benchmark._run_sweep_sync(CFG, run_sweep=fake)

    assert sweep_store.sweep_path(base, p.name).read_text() == before


def _cancelled_point(count):
    return core_sweep.SweepPoint(
        count, "cancelled", None, (), None, (12 * GIB,), 2 * GIB, "cancelled"
    )


def test_a_finished_benchmark_re_offers_the_sweep(win):
    """A fit render during a benchmark memoises the "already running" refusal,
    so the end of the benchmark thread re-offers the sweep itself."""
    _prepared(win)
    win._benchmark._benchmark_thread = object()
    win._benchmark.refresh_sweep()
    assert not win.benchmark_panel.sweep_run_btn.isEnabled()
    win._benchmark._benchmark_thread = None

    win._benchmark._on_benchmark_thread_done()

    assert win.benchmark_panel.sweep_run_btn.isEnabled()


def test_a_profile_switch_rewrites_a_typed_range(win):
    """A range typed for one profile does not survive loading another, even
    when both compute the same range."""
    _prepared(win)
    win._benchmark.refresh_sweep()
    panel = win.benchmark_panel
    panel.sweep_from.setValue(5)
    panel.sweep_to.setValue(9)
    panel.sweep_step.setValue(1)

    _prepared(win, name="Other")
    win._benchmark.refresh_sweep()

    cfg = panel.sweep_config()
    assert (cfg["start"], cfg["stop"], cfg["step"]) == (0, 8, 2)


def test_loading_a_profile_shows_its_stored_sweep(win):
    sweep_store.save(default_base_dir(), "Other", _two_point_sweep("Other"))
    _prepared(win, name="Other")

    win._benchmark.refresh_sweep()

    t = win.benchmark_panel.sweep_table
    assert [t.item(r, 0).text().split()[0] for r in range(t.rowCount())] == ["0", "4"]
    assert t.columnCount() == 7  # 5 base columns + one card + RAM
    assert win.benchmark_panel.sweep_apply_btn.isEnabled()


def test_loading_a_profile_without_a_stored_sweep_clears_the_table(win):
    sweep_store.save(default_base_dir(), "Solo", _two_point_sweep("Solo"))
    _prepared(win)
    win._benchmark.refresh_sweep()
    assert win.benchmark_panel.sweep_table.rowCount() == 2

    _prepared(win, name="Other")
    win._benchmark.refresh_sweep()

    assert win.benchmark_panel.sweep_table.rowCount() == 0
    assert not win.benchmark_panel.sweep_apply_btn.isEnabled()


def test_a_start_refusal_is_re_offered_once_the_instance_stops(win, monkeypatch):
    """A start-only refusal is not one refresh_sweep can derive, so it is not
    remembered: the next refresh re-enables Run sweep and clears the line."""
    _prepared(win)
    name = win._container_name()
    state = {"value": "running"}
    monkeypatch.setattr(
        runtime,
        "container_state",
        lambda n, b, connection="": state["value"] if n == name else "absent",
    )
    win._benchmark._run_sweep_sync(CFG, run_sweep=_stub_run_sweep({}))
    assert not win.benchmark_panel.sweep_run_btn.isEnabled()

    state["value"] = "stopped"
    win._benchmark.refresh_sweep()

    assert win.benchmark_panel.sweep_run_btn.isEnabled()
    assert win.benchmark_panel.sweep_status.text() == ""


def test_an_empty_sweep_never_replaces_the_stored_one(win, monkeypatch):
    """A cancel that lands before the first point yields a sweep with no
    points, which must not overwrite the profile's stored sweep."""
    monkeypatch.setattr(runtime, "container_state", lambda *a, **k: "absent")
    p = _prepared(win)
    base = default_base_dir()
    sweep_store.save(base, p.name, _two_point_sweep(p.name))
    before = sweep_store.sweep_path(base, p.name).read_text()

    def fake(profile, base_dir, **kwargs):
        return core_sweep.Sweep(profile.name, kwargs["knob"], "t2", (), {})

    win._benchmark._run_sweep_sync(CFG, run_sweep=fake)

    assert sweep_store.sweep_path(base, p.name).read_text() == before
