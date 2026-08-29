import dataclasses
import datetime

from PySide6.QtCore import QObject, QThread, Signal

from llama_launcher.core.spec import Profile, member_model_id
from llama_launcher.core.validation import dial_host
from llama_launcher.services import benchmark, benchmark_store
from llama_launcher.store.profiles import default_base_dir


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

    def __init__(self, client, sizes, n_predict, warmup, repeats, snapshot,
                 timestamp, parent=None):
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
                self._client, self._sizes, self._n_predict, self._warmup,
                self._repeats, self._snapshot, self._timestamp,
                should_cancel=self._should_cancel)
        except benchmark.BenchmarkError as e:
            self.failed.emit(str(e))
            return
        self.finished.emit(run)


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

        Mirrors collect_monitor_data's host/port/key/model_scope derivation
        (main_window.py, same class) exactly. Returns (client, snapshot), or
        None when there's nothing to benchmark -- a router with no model
        currently loaded, the same "not ready" condition collect_monitor_data
        treats as poll=False.
        """
        port = p.settings.get("port", 8080)
        host, key, model_scope, poll = (dial_host(p.runtime.bind_host),
                                        self.window._monitor._poll_api_key(p), None, True)
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
        still takes effect.
        """
        if run_benchmark is None:
            run_benchmark = benchmark.run_benchmark
        p = self.window._configure_panel.current_profile()
        prepared = self._prepare_benchmark(p)
        if prepared is None:
            self.window.benchmark_panel.set_benchmark_progress("No model loaded to benchmark.")
            return
        client, snapshot = prepared
        self._benchmark_profile_name = p.name
        self.window.benchmark_panel.set_benchmark_running(True)
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        try:
            run = run_benchmark(client, cfg["sizes"], cfg["n_predict"], cfg["warmup"],
                                cfg["repeats"], snapshot, timestamp)
        except benchmark.BenchmarkError as e:
            self._on_benchmark_failed(str(e))
            return
        self._on_benchmark_finished(run)

    def _on_benchmark_run(self, cfg: dict) -> None:
        """Production path: build the endpoint/client on the UI thread, then run
        benchmark.run_benchmark() on a QThread so a slow benchmark never blocks
        the GUI."""
        if self._benchmark_thread is not None:
            return          # a run is already active; the panel showed Cancel
        p = self.window._configure_panel.current_profile()
        prepared = self._prepare_benchmark(p)
        if prepared is None:
            self.window.benchmark_panel.set_benchmark_progress("No model loaded to benchmark.")
            return
        client, snapshot = prepared
        self._benchmark_profile_name = p.name
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")

        thread = QThread(self.window)
        worker = BenchmarkWorker(client, cfg["sizes"], cfg["n_predict"], cfg["warmup"],
                                 cfg["repeats"], snapshot, timestamp)
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
        """Release the finished thread/worker so a later run can start."""
        worker, thread = self._benchmark_worker, self._benchmark_thread
        self._benchmark_worker = None
        self._benchmark_thread = None
        if worker is not None:
            worker.deleteLater()
        if thread is not None:
            thread.deleteLater()

    def _on_benchmark_cancel(self) -> None:
        if self._benchmark_worker is not None:
            self._benchmark_worker.cancel()

    def _on_benchmark_clear(self) -> None:
        """Wipe the saved benchmark history for the current profile and the view."""
        benchmark_store.clear(default_base_dir(), self.window._configure_panel.current_profile().name)
        self.window.benchmark_panel.reset()

    def _on_benchmark_finished(self, run) -> None:
        name = self._benchmark_profile_name or self.window._configure_panel.current_profile().name
        base = default_base_dir()
        previous_runs = benchmark_store.load(base, name)
        previous = previous_runs[-1] if previous_runs else None
        benchmark_store.append(base, name, run)
        run_dict = dataclasses.asdict(run)
        delta = benchmark_store.delta(run_dict, previous) if previous is not None else None
        self.window.benchmark_panel.show_benchmark_run(run_dict, delta)
        self.window.benchmark_panel.set_benchmark_history(benchmark_store.load(base, name))
        self.window.benchmark_panel.set_benchmark_running(False)

    def _on_benchmark_failed(self, msg: str) -> None:
        self.window.benchmark_panel.set_benchmark_progress(f"Benchmark failed: {msg}")
        self.window.benchmark_panel.set_benchmark_running(False)

    # -- teardown -------------------------------------------------------------
    def drain(self) -> None:
        """Tear down a running benchmark thread: cancel the worker (so a loop
        mid-repeat unwinds instead of running to completion against a closed
        window) and wait for the QThread to actually stop, since a Python
        interpreter shutdown with a live QThread can abort/crash.

        worker.finished/failed are already wired to thread.quit() (see
        _on_benchmark_run), but that's a queued cross-thread connection: it
        only takes effect once THIS (UI) thread's event loop runs and
        delivers it. A bare wait() never pumps events, so it would deadlock
        waiting for a quit() that never arrives; pump events between short
        waits instead. terminate() is a last-resort backstop so a stuck
        worker can never block window/app teardown indefinitely.
        """
        thread = getattr(self, "_benchmark_thread", None)
        if thread is None:
            return
        worker = getattr(self, "_benchmark_worker", None)
        if worker is not None:
            worker.cancel()
        from PySide6.QtCore import QCoreApplication
        for _ in range(100):        # ~2s ceiling
            if thread.wait(20):
                # One more pump so the already-queued finished/failed ->
                # _on_benchmark_thread_done delivery (which clears
                # _benchmark_thread/_benchmark_worker) actually runs before
                # we return, instead of leaving it dangling until whatever
                # next processes the event queue.
                QCoreApplication.processEvents()
                return
            QCoreApplication.processEvents()
        thread.terminate()
        thread.wait(100)
