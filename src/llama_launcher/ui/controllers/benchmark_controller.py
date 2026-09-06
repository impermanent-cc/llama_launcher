import dataclasses
import datetime
import subprocess

from PySide6.QtCore import QObject, QThread, Signal

from llama_launcher.core import command_builder, memory_fit, vram
from llama_launcher.core import sweep as core_sweep
from llama_launcher.core.settings_catalog import CATALOG, accepts
from llama_launcher.core.spec import Profile, member_model_id, profile_port
from llama_launcher.core.validation import dial_host
from llama_launcher.services import (
    benchmark,
    benchmark_store,
    headless,
    runtime,
    sweep_store,
)
from llama_launcher.services import sweep as sweep_service
from llama_launcher.store.profiles import default_base_dir

_ESTIMATE_KEYS = (
    "engine",
    "free_bytes_per_gpu",
    "raw_args",
    "draft_meta",
    "draft_weights",
    "mmproj_bytes",
)


class BenchmarkWorker(QObject):
    """Runs benchmark.run_benchmark() off the UI thread.

    Built with an already-constructed client/snapshot/timestamp (endpoint
    derivation and profile reads happen on the UI thread, before this worker
    is started) so run() touches no GUI/profile state -- it only calls
    run_benchmark() and emits a result signal. Qt delivers finished/failed to
    the UI-thread slot via a queued connection since this object lives on a
    different thread than MainWindow.
    """

    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        client,
        sizes,
        n_predict,
        warmup,
        repeats,
        snapshot,
        timestamp,
        parent=None,
    ):
        super().__init__(parent)
        self._client = client
        self._sizes = sizes
        self._n_predict = n_predict
        self._warmup = warmup
        self._repeats = repeats
        self._snapshot = snapshot
        self._timestamp = timestamp
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _should_cancel(self) -> bool:
        return self._cancelled

    def run(self) -> None:
        try:
            run = benchmark.run_benchmark(
                self._client,
                self._sizes,
                self._n_predict,
                self._warmup,
                self._repeats,
                self._snapshot,
                self._timestamp,
                should_cancel=self._should_cancel,
            )
        except benchmark.BenchmarkError as e:
            self.failed.emit(str(e))
            return
        self.finished.emit(run)


class SweepWorker(QObject):
    """Runs services.sweep.run_sweep() off the UI thread.

    Built with a profile, a base dir and the run_sweep keyword arguments the
    controller assembled on the UI thread (including the estimate_for
    closure, which captures its widget values there), so run() touches no
    GUI state. Points are emitted as plain dicts because Qt delivers them to
    the UI thread through a queued connection.
    """

    point = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, profile, base_dir, kwargs, parent=None):
        super().__init__(parent)
        self._profile = profile
        self._base_dir = base_dir
        self._kwargs = kwargs
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _should_cancel(self) -> bool:
        return self._cancelled

    def run(self) -> None:
        try:
            sweep = sweep_service.run_sweep(
                self._profile,
                self._base_dir,
                should_cancel=self._should_cancel,
                on_point=lambda pt: self.point.emit(dataclasses.asdict(pt)),
                **self._kwargs,
            )
        except Exception as e:  # a sweep must never take the window down
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.finished.emit(sweep)


class BenchmarkController:
    """Owns the benchmark run lifecycle and its worker/thread.

    Widgets stay on the window (built by MainWindow -- `benchmark_panel`);
    this controller only owns behavior plus the plain state below
    (_benchmark_thread/_benchmark_worker/_benchmark_profile_name). Members
    this controller itself owns (e.g. `_prepare_benchmark`,
    `_resolve_benchmark_member`, `_on_benchmark_finished`) are called directly
    as `self.<method>(...)`; widgets and methods owned by other
    panels/controllers go through `self.window._<owner>.<x>` (e.g.
    `self.window._configure_panel.current_profile()`,
    `self.window._monitor._poll_api_key`).
    """

    def __init__(self, window):
        self.window = window

        self._benchmark_thread = None
        self._benchmark_worker = None
        self._benchmark_profile_name = None

        self._sweep_thread = None
        self._sweep_worker = None
        self._sweep_knob_used = None
        self._sweep_n_cards = 0
        # (profile, knob) of the sweep in flight: what the teardown needs to
        # name the sweep container after a terminated worker.
        self._sweep_planned = None
        self._sweep_result_profile = None
        # What the panel was last told: the (profile, knob, from, to, step)
        # prefill and the (profile, reason) availability. A refresh writes
        # only what changed, and a profile switch always changes both.
        self._sweep_prefill_shown = None
        self._sweep_reason_shown = None
        # (profile, card count) the sweep table currently shows.
        self._sweep_shown = None

    def _resolve_benchmark_member(self, p: Profile, model_scope: str | None):
        """The child Profile for a router's loaded model, for build_snapshot().

        build_snapshot() requires a Profile (it reads .settings/.model), never
        a RouterMember -- a RouterMember has neither and raises AttributeError.
        Falling back to None makes build_snapshot record blank config fields
        for a router (its own settings are leftover form state, not what the
        member ran with) -- a lesser-quality snapshot but never a wrong one.
        """
        if p.mode != "router" or model_scope is None:
            return None
        for member, member_profile in self.window._configure_panel.member_pairs():
            if member_model_id(member) == model_scope:
                return member_profile
        return None

    def _prepare_benchmark(self, p: Profile):
        """Endpoint + snapshot + client for a benchmark run.

        Derives host/port/key/model_scope the same way collect_monitor_data
        does. Returns (client, snapshot), or None when there's nothing to
        benchmark: a router with no model currently loaded, the same "not
        ready" condition collect_monitor_data treats as poll=False.
        """
        port = profile_port(p)
        host, key, model_scope, poll = (
            dial_host(p.runtime.bind_host),
            self.window._monitor._poll_api_key(p),
            None,
            True,
        )
        if p.mode == "router":
            host = self.window._monitor._router_host(p)
            model_scope = self.window._monitor._router_pollable_model()
            poll = model_scope is not None
        if not poll:
            return None
        member = self._resolve_benchmark_member(p, model_scope)
        snapshot = benchmark.build_snapshot(p, member=member)
        client = benchmark.requests_client(host, port, key, model_scope)
        return client, snapshot

    def _run_benchmark_sync(self, cfg: dict, run_benchmark=None) -> None:
        """Endpoint-build -> run_benchmark -> save/show, all on the calling thread.

        The testable seam: tests call this directly with a stubbed
        run_benchmark, so no QThread and no network round-trip is needed.
        `run_benchmark` resolves benchmark.run_benchmark dynamically when
        omitted (rather than as a plain default-argument value), so
        monkeypatching the module attribute after this method is defined
        still takes effect. A sweep in flight refuses the run: it holds the
        profile's port with its own container.
        """
        if self._sweep_thread is not None:
            self.window.benchmark_panel.set_benchmark_progress("A sweep is running.")
            return
        if run_benchmark is None:
            run_benchmark = benchmark.run_benchmark
        p = self.window._configure_panel.current_profile()
        prepared = self._prepare_benchmark(p)
        if prepared is None:
            self.window.benchmark_panel.set_benchmark_progress(
                "No model loaded to benchmark."
            )
            return
        client, snapshot = prepared
        self._benchmark_profile_name = p.name
        self.window.benchmark_panel.set_benchmark_running(True)
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        try:
            run = run_benchmark(
                client,
                cfg["sizes"],
                cfg["n_predict"],
                cfg["warmup"],
                cfg["repeats"],
                snapshot,
                timestamp,
            )
        except benchmark.BenchmarkError as e:
            self._on_benchmark_failed(str(e))
            return
        self._on_benchmark_finished(run)

    def _on_benchmark_run(self, cfg: dict) -> None:
        """Production path: build the endpoint/client on the UI thread, then run
        benchmark.run_benchmark() on a QThread so a slow benchmark never blocks
        the GUI. A sweep in flight refuses the run: it holds the profile's
        port with its own container."""
        if self._sweep_thread is not None:
            self.window.benchmark_panel.set_benchmark_progress("A sweep is running.")
            return
        if self._benchmark_thread is not None:
            return  # a run is already active; the panel showed Cancel
        p = self.window._configure_panel.current_profile()
        prepared = self._prepare_benchmark(p)
        if prepared is None:
            self.window.benchmark_panel.set_benchmark_progress(
                "No model loaded to benchmark."
            )
            return
        client, snapshot = prepared
        self._benchmark_profile_name = p.name
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")

        thread = QThread(self.window)
        worker = BenchmarkWorker(
            client,
            cfg["sizes"],
            cfg["n_predict"],
            cfg["warmup"],
            cfg["repeats"],
            snapshot,
            timestamp,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # Queued across threads by Qt automatically -- worker lives on `thread`,
        # these slots run on the UI thread where GUI/store access is safe.
        worker.finished.connect(self._on_benchmark_finished)
        worker.failed.connect(self._on_benchmark_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._on_benchmark_thread_done)
        self._benchmark_thread = thread
        self._benchmark_worker = worker
        self.window.benchmark_panel.set_benchmark_running(True)
        thread.start()

    def _on_benchmark_thread_done(self) -> None:
        """Release the finished thread/worker so a later run can start, and
        re-offer the sweep: the refusal the panel holds was written while this
        run was in flight, and a refresh alone would keep it (the memo still
        matches)."""
        worker, thread = self._benchmark_worker, self._benchmark_thread
        self._benchmark_worker = None
        self._benchmark_thread = None
        if worker is not None:
            worker.deleteLater()
        if thread is not None:
            thread.deleteLater()
        self._sweep_reason_shown = None
        self.refresh_sweep()

    def _on_benchmark_cancel(self) -> None:
        if self._benchmark_worker is not None:
            self._benchmark_worker.cancel()

    def _on_benchmark_clear(self) -> None:
        """Wipe the saved benchmark history for the current profile and the view."""
        benchmark_store.clear(
            default_base_dir(), self.window._configure_panel.current_profile().name
        )
        self.window.benchmark_panel.reset()

    def _on_benchmark_finished(self, run) -> None:
        name = (
            self._benchmark_profile_name
            or self.window._configure_panel.current_profile().name
        )
        base = default_base_dir()
        previous_runs = benchmark_store.load(base, name)
        previous = previous_runs[-1] if previous_runs else None
        benchmark_store.append(base, name, run)
        run_dict = dataclasses.asdict(run)
        delta = (
            benchmark_store.delta(run_dict, previous) if previous is not None else None
        )
        self.window.benchmark_panel.show_benchmark_run(run_dict, delta)
        self.window.benchmark_panel.set_benchmark_history(
            benchmark_store.load(base, name)
        )
        self.window.benchmark_panel.set_benchmark_running(False)

    def _on_benchmark_failed(self, msg: str) -> None:
        self.window.benchmark_panel.set_benchmark_progress(f"Benchmark failed: {msg}")
        self.window.benchmark_panel.set_benchmark_running(False)

    # -- offload sweep --------------------------------------------------------
    def _sweep_availability(self, p) -> str:
        """The one-line reason this profile cannot be swept, empty when it
        can: only a local container profile with a model, and only while no
        benchmark or sweep is running.

        Cheap by contract: it reads the profile and this controller's own
        state only, so the panel can re-derive it on every fit render. The
        probes that cost a subprocess live in _sweep_start_refusal.
        """
        if p.mode == "router":
            return "Sweep a member profile: a router has no model of its own."
        if p.runtime.launch_mode != "container":
            return (
                "Sweep needs a container profile; native and RPC launches "
                "are not driven headlessly."
            )
        if (p.runtime.node or "local") != "local":
            return "Sweep runs local containers only in this version."
        if not p.model:
            return "Select a model first."
        knob = self._sweep_knob()
        if not accepts(CATALOG[knob], p.runtime.engine):
            return (
                f"{p.runtime.engine} has no --{knob}: sweep a MoE model or "
                "use llama.cpp."
            )
        if self._benchmark_thread is not None or self._sweep_thread is not None:
            return "A benchmark or sweep is already running."
        return ""

    def _sweep_start_refusal(self, p) -> str:
        """_sweep_availability plus the checks only a start can afford: the
        container-state probe (a subprocess) and the panel's prompt sizes."""
        reason = self._sweep_availability(p)
        if reason:
            return reason
        if (
            runtime.container_state(self.window._container_name(), p.runtime.binary)
            == "running"
        ):
            return "Stop the running instance first: the sweep uses the same port."
        if not self._bench_cfg()["sizes"]:
            return "Set at least one prompt size in the benchmark row first."
        knob = self._sweep_knob()
        if knob in command_builder.raw_arg_values(p.raw_args):
            return (
                f"Remove --{knob} from the raw arguments first: it overrides "
                "the swept count."
            )
        sweep_name = headless._container_name(sweep_service.sweep_profile(p, 0, knob))
        if runtime.container_state(sweep_name, p.runtime.binary) == "running":
            return (
                f"A previous sweep container is still running: stop {sweep_name} first."
            )
        return ""

    def refresh_sweep(self) -> None:
        """Availability and prefill together, for the moments the loaded
        profile or its fit readout changes.

        Both writes are idempotent by memory: the panel is written only when
        the reason or the computed range actually changes, so a debounced fit
        render cannot clobber a range the user typed or a status the sweep
        left. Each memory carries the profile name, so a switch to another
        profile always rewrites both. A sweep in flight owns both lines, so a
        refresh writes nothing.
        """
        if self._sweep_thread is not None:
            return
        p = self.window._configure_panel.current_profile()
        reason = self._sweep_availability(p)
        if (p.name, reason) != self._sweep_reason_shown:
            self._sweep_reason_shown = (p.name, reason)
            # The panel writes its status line only while refusing, so
            # refusing with the new reason (empty when allowed) is what
            # clears a stale one.
            self.window.benchmark_panel.set_sweep_available(False, reason)
            if not reason:
                self.window.benchmark_panel.set_sweep_available(True)
        self._show_stored_sweep(p)
        self.sweep_prefill()

    def _show_stored_sweep(self, p) -> None:
        """Repaint the sweep table from this profile's stored sweep, or empty
        it when the profile has none, so the table never shows another
        profile's points.

        Written only when the table holds another profile or a stale card
        count: the first render after a profile is loaded often has no card
        figures yet, and the one that brings them must widen the table.
        """
        fit_kwargs = self.window._configure_panel._fit_report_kwargs()
        n_cards = len(fit_kwargs["free_bytes_per_gpu"])
        if (p.name, n_cards) == self._sweep_shown:
            return
        self._sweep_shown = (p.name, n_cards)
        stored = sweep_store.load(default_base_dir(), p.name)
        if not stored:
            self._sweep_result_profile = None
            self.window.benchmark_panel.clear_sweep()
            return
        # Apply targets the stored sweep's own knob, which need not be the one
        # the current metadata would choose.
        self._sweep_knob_used = stored.get("knob") or self._sweep_knob()
        self._sweep_result_profile = p.name
        self._sweep_n_cards = n_cards
        self.window.benchmark_panel.show_sweep(stored, n_cards)

    def _sweep_knob(self) -> str:
        """--n-cpu-moe on a MoE model, --n-cpu-ffn otherwise; a model whose
        metadata has not been read counts as dense."""
        meta = self.window._configure_panel._fit_meta
        return core_sweep.sweep_knob(meta is not None and memory_fit.is_moe(meta))

    def sweep_prefill(self) -> None:
        """Set the panel's knob label and from/to/step from the current
        estimate: from the smallest offload count that fits when the model
        does not fit on that knob, else 0.

        Writes the panel only when the computed (profile, knob, from, to,
        step) differs from the last one written, so a range the user typed
        stands until the profile or its estimate moves it.
        """
        panel = self.window._configure_panel
        name = panel._profile_name()
        knob = self._sweep_knob()
        report = panel._current_fit_report()
        smallest = None
        if report is not None and not report.fits:
            kwargs = panel._fit_report_kwargs()
            kwargs.pop("ram_available", None)
            meta = kwargs.pop("meta")
            weights = kwargs.pop("weights_bytes")
            found = memory_fit.smallest_fitting_offload(meta, weights, **kwargs)
            if found is not None and found[0] == knob:
                smallest = found[1]
        start, stop, step = core_sweep.default_range(smallest)
        if (name, knob, start, stop, step) == self._sweep_prefill_shown:
            return
        self._sweep_prefill_shown = (name, knob, start, stop, step)
        self.window.benchmark_panel.set_sweep_prefill(knob, start, stop, step)

    def _bench_cfg(self) -> dict:
        """The Benchmark panel's own prompt sizes, n-predict, warmup and
        repeats: every sweep point is benchmarked with them."""
        panel = self.window.benchmark_panel
        try:
            sizes = [int(s) for s in panel.bench_sizes.text().split(",") if s.strip()]
        except ValueError:
            sizes = []
        return {
            "sizes": sizes,
            "n_predict": panel.bench_npredict.value(),
            "warmup": panel.bench_warmup.value(),
            "repeats": panel.bench_repeats.value(),
        }

    @staticmethod
    def _estimate_closure(knob: str, fit_kwargs: dict):
        """estimate_for(count) -> (per-card estimated bytes, estimated RAM)
        for the profile's settings with `count` laid over the knob, computed
        from values already read on the UI thread so the sweep worker touches
        no widget. ((), 0) when the metadata cannot support an estimate.

        A card figure is weights + KV + compute, excluding the fixed card
        overhead, so it names the same buffers as the measured figure the
        sweep reads back from the server's load log.
        """
        meta = fit_kwargs["meta"]
        weights = fit_kwargs["weights_bytes"]
        settings = dict(fit_kwargs["settings"])
        rest = {k: fit_kwargs[k] for k in _ESTIMATE_KEYS}

        def estimate_for(count):
            est = vram.estimate_memory(
                meta, weights, settings={**settings, knob: int(count)}, **rest
            )
            if est is None:
                return (), 0
            cards = tuple(c.weights + c.kv + c.compute for c in est.cards)
            return cards, est.ram.total

        return estimate_for

    def _sweep_plan(self, cfg: dict):
        """(profile, run_sweep keyword arguments) for one sweep, or None when
        the profile is refused and the panel shows why. Every widget read for
        the run happens here, on the UI thread."""
        p = self.window._configure_panel.current_profile()
        reason = self._sweep_start_refusal(p)
        if reason:
            self.window.benchmark_panel.set_sweep_available(False, reason)
            # This refusal is the start-only one, which refresh_sweep never
            # derives; dropping the memory lets the next refresh re-offer the
            # sweep once the condition clears.
            self._sweep_reason_shown = None
            return None
        fit_kwargs = self.window._configure_panel._fit_report_kwargs()
        knob = self._sweep_knob()
        self._sweep_knob_used = knob
        self._sweep_planned = (p, knob)
        self._sweep_n_cards = len(fit_kwargs["free_bytes_per_gpu"])
        self._sweep_shown = (p.name, self._sweep_n_cards)
        # Empty the table for the planned card count so the per-point updates
        # land under the right headers instead of the previous sweep's.
        self.window.benchmark_panel.show_sweep({"points": []}, self._sweep_n_cards)
        kwargs = dict(
            knob=knob,
            counts=core_sweep.sweep_counts(
                cfg["start"],
                cfg["stop"],
                cfg["step"],
                getattr(fit_kwargs.get("meta"), "n_layers", 0),
            ),
            bench_cfg=self._bench_cfg(),
            estimate_for=self._estimate_closure(knob, fit_kwargs),
            ready_timeout=cfg["ready_timeout"],
        )
        return p, kwargs

    def _run_sweep_sync(self, cfg: dict, run_sweep=None) -> None:
        """Plan -> run_sweep -> save/show, all on the calling thread.

        The seam that runs a whole sweep without a QThread or a container: a
        caller passes its own `run_sweep`. `run_sweep` resolves
        services.sweep.run_sweep dynamically when omitted, so monkeypatching
        the module attribute still takes effect.
        """
        if run_sweep is None:
            run_sweep = sweep_service.run_sweep
        plan = self._sweep_plan(cfg)
        if plan is None:
            return
        p, kwargs = plan
        self.window.benchmark_panel.set_sweep_running(True)
        try:
            sweep = run_sweep(
                p,
                default_base_dir(),
                on_point=lambda pt: self._on_sweep_point(dataclasses.asdict(pt)),
                **kwargs,
            )
        except Exception as e:
            self._on_sweep_failed(f"{type(e).__name__}: {e}")
            return
        self._on_sweep_finished(sweep)

    def _on_sweep_run(self, cfg: dict) -> None:
        """Production path: plan on the UI thread, then run the sweep on a
        QThread so the restarts and benchmarks never block the GUI."""
        if self._sweep_thread is not None:
            return  # a sweep is already active; the panel showed Cancel
        plan = self._sweep_plan(cfg)
        if plan is None:
            return
        p, kwargs = plan

        thread = QThread(self.window)
        worker = SweepWorker(p, default_base_dir(), kwargs)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # Queued across threads by Qt automatically: the worker lives on
        # `thread`, so these slots run on the UI thread where GUI and store
        # access is safe.
        worker.point.connect(self._on_sweep_point)
        worker.finished.connect(self._on_sweep_finished)
        worker.failed.connect(self._on_sweep_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._on_sweep_thread_done)
        self._sweep_thread = thread
        self._sweep_worker = worker
        self.window.benchmark_panel.set_sweep_running(True)
        thread.start()

    def _on_sweep_thread_done(self) -> None:
        """Release the finished thread/worker so a later sweep can start."""
        worker, thread = self._sweep_worker, self._sweep_thread
        self._sweep_worker = None
        self._sweep_thread = None
        if worker is not None:
            worker.deleteLater()
        if thread is not None:
            thread.deleteLater()

    def _on_sweep_cancel(self) -> None:
        if self._sweep_worker is not None:
            self._sweep_worker.cancel()

    def _on_sweep_point(self, point: dict) -> None:
        self.window.benchmark_panel.set_sweep_point(point)

    def _on_sweep_finished(self, sweep) -> None:
        """Store the sweep beside the benchmark history, which it never
        touches, and repaint the sweep table from it.

        A cancelled sweep, and one cancelled before its first point ran, is
        shown but never stored: its points stop wherever the cancel landed, so
        saving it would replace a complete stored sweep with a truncated or
        empty one.
        """
        cancelled = any(pt.status == "cancelled" for pt in sweep.points)
        if not cancelled and sweep.points:
            sweep_store.save(default_base_dir(), sweep.profile, sweep)
        self._sweep_knob_used = sweep.knob
        self._sweep_result_profile = sweep.profile
        self._sweep_shown = (sweep.profile, self._sweep_n_cards)
        self._sweep_planned = None
        self.window.benchmark_panel.show_sweep(
            dataclasses.asdict(sweep), self._sweep_n_cards
        )
        self.window.benchmark_panel.set_sweep_running(False)
        if cancelled:
            self.window.benchmark_panel.set_sweep_available(True, "Sweep cancelled.")

    def _on_sweep_failed(self, msg: str) -> None:
        self._sweep_planned = None
        self.window.benchmark_panel.set_sweep_available(True, f"Sweep failed: {msg}")
        self.window.benchmark_panel.set_sweep_running(False)

    def _on_sweep_apply(self, count: int) -> None:
        """Write the winning count into the Configure form's offload knob and
        save the profile; nothing else about the profile changes. It writes
        only while the form still holds the profile the sweep ran, so Apply
        can never save a count into a different model's profile."""
        panel = self.window.benchmark_panel
        swept = self._sweep_result_profile
        if swept is None:
            panel.set_sweep_available(True, "Run a sweep before Apply.")
            return
        if self.window._configure_panel._profile_name() != swept:
            panel.set_sweep_available(
                True,
                f"Load the swept profile '{swept}' in Configure before Apply.",
            )
            return
        knob = self._sweep_knob_used or self._sweep_knob()
        widget = self.window._configure_panel._widgets.get(knob)
        if widget is None:
            return
        widget.set_value(int(count))
        self.window.save_current_profile()

    # -- teardown -------------------------------------------------------------
    def drain(self) -> None:
        """Tear down a running benchmark thread and a running sweep thread:
        cancel the worker (so a loop mid-repeat or mid-point unwinds instead
        of running to completion against a closed window) and wait for the
        QThread to actually stop, since a Python interpreter shutdown with a
        live QThread can abort/crash.
        """
        self._drain_thread(
            getattr(self, "_benchmark_worker", None),
            getattr(self, "_benchmark_thread", None),
        )
        terminated = self._drain_thread(
            getattr(self, "_sweep_worker", None), getattr(self, "_sweep_thread", None)
        )
        if terminated:
            self._remove_sweep_container()

    def _remove_sweep_container(self) -> None:
        """Remove the sweep's container without waiting for it.

        run_sweep removes it in its own finally, which a terminated worker
        never reaches, so without this the point in flight would keep the
        model on the GPU after the window is gone. `rm -f` kills and removes
        in one command, spawned detached so a window closing never blocks the
        UI thread on podman. The container name is derived from the profile
        and the knob alone, so any count names it.
        """
        planned = getattr(self, "_sweep_planned", None)
        if planned is None:
            return
        p, knob = planned
        name = headless._container_name(sweep_service.sweep_profile(p, 0, knob))
        try:
            subprocess.Popen(
                runtime.rm_argv(name, p.runtime.binary),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            pass
        self._sweep_planned = None

    @staticmethod
    def _drain_thread(worker, thread) -> bool:
        """Cancel the given worker and wait out its thread (the benchmark
        pair or the sweep pair); True when the wait ran out and the thread
        had to be terminated.

        worker.finished/failed are already wired to thread.quit() where the
        pair is built, but that's a queued cross-thread connection: it
        only takes effect once THIS (UI) thread's event loop runs and
        delivers it. A bare wait() never pumps events, so it would deadlock
        waiting for a quit() that never arrives; pump events between short
        waits instead. terminate() is a last-resort backstop so a stuck
        worker can never block window/app teardown indefinitely.
        """
        if thread is None:
            return False
        if worker is not None:
            worker.cancel()
        from PySide6.QtCore import QCoreApplication

        for _ in range(100):  # ~2s ceiling
            if thread.wait(20):
                # One more pump so the already-queued finished/failed ->
                # thread-done delivery (which clears the pair's thread and
                # worker) actually runs before we return, instead of leaving
                # it dangling until whatever next processes the event queue.
                QCoreApplication.processEvents()
                return False
            QCoreApplication.processEvents()
        thread.terminate()
        thread.wait(100)
        return True
