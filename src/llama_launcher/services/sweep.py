"""Run an offload sweep: for each count, launch the profile as a detached
container, wait for the server, read its load-time memory lines, run the
benchmark, then stop and remove the container."""

import copy
import datetime
import functools
import subprocess
import time

from llama_launcher.core import sweep as core_sweep
from llama_launcher.core.settings_catalog import CATALOG, accepts
from llama_launcher.core.spec import profile_port
from llama_launcher.core.validation import dial_host
from llama_launcher.services import benchmark, headless, runtime
from llama_launcher.services.health import probe_health

_MIN_SWEEP_VERBOSITY = 4


def sweep_profile(profile, count: int, knob: str):
    """A copy of the profile named for the sweep container, with the count
    laid over the knob. Verbosity is raised to the level that prints
    load-time buffer-size lines, unless the profile already asks for more."""
    q = copy.deepcopy(profile)
    q.name = f"{profile.name} sweep"
    q.settings[knob] = int(count)
    if accepts(CATALOG["verbosity"], q.runtime.engine):
        current = q.settings.get("verbosity")
        if current is None or int(current) < _MIN_SWEEP_VERBOSITY:
            q.settings["verbosity"] = _MIN_SWEEP_VERBOSITY
    return q


def _wait_ready(host, port, timeout, should_cancel, *, name=None, binary=None) -> bool:
    """Poll the server until it reports ready. A container that is no longer
    running gives up at once instead of burning the whole timeout, so a
    server that dies while loading costs one poll interval."""
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if should_cancel is not None and should_cancel():
            return False
        if probe_health(port, host=host) == "ready":
            return True
        if name and runtime.container_state(name, binary) != "running":
            return False
        time.sleep(1.0)
    return False


def _read_log(name, binary) -> str:
    try:
        res = subprocess.run(
            runtime.logs_once_argv(name, binary),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (res.stdout or "") + (res.stderr or "")


def _stop_and_remove(name, binary, timeout) -> None:
    for argv in (
        runtime.stop_argv(name, binary, timeout),
        runtime.rm_argv(name, binary),
    ):
        try:
            subprocess.run(
                argv, capture_output=True, text=True, check=False, timeout=timeout + 30
            )
        except (OSError, subprocess.SubprocessError):
            pass


def run_sweep(
    profile,
    base_dir,
    *,
    knob,
    counts,
    bench_cfg,
    estimate_for,
    ready_timeout,
    should_cancel=None,
    on_point=None,
    launch=None,
    wait_ready=None,
    read_log=None,
    run_benchmark=None,
    stop=None,
    clock=None,
):
    """One sweep over `counts`. Every probe is a keyword so a test can run
    the loop without podman or a network. `estimate_for(count)` returns
    (per-card estimated bytes for model plus KV plus compute, estimated RAM)
    for that count."""
    launch = launch or headless.launch
    read_log = read_log or _read_log
    run_benchmark = run_benchmark or benchmark.run_benchmark
    stop = stop or _stop_and_remove
    clock = clock or time.monotonic
    binary = profile.runtime.binary
    points = []
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    for count in counts:
        if should_cancel is not None and should_cancel():
            break
        p = sweep_profile(profile, count, knob)
        name = headless._container_name(p)
        est_cards, est_ram = estimate_for(count)
        point = core_sweep.SweepPoint(
            count, "running", None, (), None, tuple(est_cards), int(est_ram), ""
        )
        if on_point is not None:
            on_point(point)
        started = clock()
        launched = False
        # The real waiter also watches the container, which only run_sweep can
        # name; an injected waiter keeps the four positional arguments.
        waiter = wait_ready or functools.partial(_wait_ready, name=name, binary=binary)
        try:
            result = launch(p, base_dir, binary)
            if not result.ok:
                point = _failed(point, result.error or "launch failed")
            else:
                launched = True
                host = dial_host(p.runtime.bind_host)
                port = profile_port(p)
                ready = waiter(host, port, ready_timeout, should_cancel)
                ready_seconds = round(clock() - started, 1)
                log = read_log(name, binary)
                if not ready:
                    point = _failed(
                        point, core_sweep.last_log_line(log) or "not ready in time"
                    )
                else:
                    measured = core_sweep.parse_load_log(log)
                    client = benchmark.requests_client(
                        host, port, p.settings.get("api-key") or "", None
                    )
                    run = run_benchmark(
                        client,
                        bench_cfg["sizes"],
                        bench_cfg["n_predict"],
                        bench_cfg["warmup"],
                        bench_cfg["repeats"],
                        benchmark.build_snapshot(p),
                        timestamp,
                        should_cancel,
                    )
                    rows = tuple(vars(r) for r in run.rows)
                    point = core_sweep.SweepPoint(
                        count,
                        "ok",
                        ready_seconds,
                        rows,
                        measured,
                        point.estimated_cards,
                        point.estimated_ram,
                        "",
                    )
        except benchmark.BenchmarkError as e:
            point = _failed(point, str(e))
        except Exception as e:
            point = _failed(point, f"{type(e).__name__}: {e}")
        finally:
            # Only a launch that took the name may stop and remove it: a
            # refused launch leaves whatever already answers to that name
            # alone.
            if launched:
                stop(name, binary, p.runtime.stop_timeout)
        cancelled = should_cancel is not None and should_cancel()
        if cancelled:
            point = _cancelled(point)
        points.append(point)
        if on_point is not None:
            on_point(point)
        if cancelled:
            break
    return core_sweep.Sweep(
        profile.name, knob, timestamp, tuple(points), dict(bench_cfg)
    )


def _cancelled(point):
    """The point in flight when the sweep was cancelled: neither a result nor
    a failure of the profile."""
    return core_sweep.SweepPoint(
        point.count,
        "cancelled",
        None,
        (),
        None,
        point.estimated_cards,
        point.estimated_ram,
        "cancelled",
    )


def _failed(point, error):
    return core_sweep.SweepPoint(
        point.count,
        "failed",
        None,
        (),
        None,
        point.estimated_cards,
        point.estimated_ram,
        error,
    )
