import datetime
import time

from PySide6.QtCore import QRunnable, QThread, QThreadPool, QTimer, Signal

from llama_launcher.core.spec import DEFAULT_STOP_TIMEOUT, Profile, Runtime
from llama_launcher.core.instances import build_instances, worker_card_title
from llama_launcher.core.mtp_stats import spec_counters, spec_delta
from llama_launcher.core.nodes import connection_for, host_of
from llama_launcher.core.validation import dial_host
from llama_launcher.store.nodes import load_nodes, get_node
from llama_launcher.store.profiles import list_profiles, load_config, save_config
from llama_launcher.services import runtime, health, metrics, gpu, native, rpc
from llama_launcher.services import api_key as api_key_store
from llama_launcher.services import router_api


def _sigkill_if_alive(pid: int) -> None:
    """SIGKILL `pid` only if it is still OUR native process -- the guard against
    force-killing a pid the OS has recycled to an unrelated process during the
    grace window. The binary is read from the still-live registry entry (a
    running native process is never pruned, so if the entry is gone the process
    is already dead and no kill is needed)."""
    import signal as _signal
    from llama_launcher.ui.main_window import base_dir

    entry = next((e for e in native.read_entries(base_dir())
                  if e.get("pid") == pid), None)
    if entry is not None and native.is_alive(pid, entry.get("binary", "")):
        native.stop_native(pid, _signal.SIGKILL)


def _schedule_sigkill(pid: int, delay: int) -> None:
    """Send SIGKILL after `delay`s if the process is still alive -- the native
    analog of `podman stop -t`. SIGTERM was already sent; give it the grace
    period, then force (guarded against a recycled pid, see _sigkill_if_alive)."""
    from PySide6.QtCore import QTimer
    QTimer.singleShot(max(0, delay) * 1000, lambda: _sigkill_if_alive(pid))


def _fmt_uptime(started_at: str | None) -> str:
    """Return a short human-readable uptime string from a started_at ISO timestamp."""
    if not started_at:
        return ""
    try:
        dt = datetime.datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        elapsed = int((now - dt).total_seconds())
        if elapsed < 60:
            return f"{elapsed}s"
        if elapsed < 3600:
            return f"{elapsed // 60}m"
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        return f"{h}h{m:02d}m"
    except (ValueError, TypeError):
        return ""


def focused_gpu_ssh(node_name: str, base_dir) -> str:
    """SSH target for the node a monitored instance runs on ('' for local/missing)."""
    from llama_launcher.store.nodes import gpu_ssh_target
    return gpu_ssh_target(base_dir, node_name)


def build_monitor_data(target: dict) -> dict | None:
    """Gather the Monitor summary from a primitives-only `target`.

    Pure by design: it touches no widget/profile state, only the plain values
    the UI thread snapshotted into `target` (GIL-safe to read from a worker).
    This is the blocking part of the poll -- podman stats, nvidia-smi and the
    /metrics + /slots HTTP calls -- so a worker runs it off the UI thread while
    update_status keeps the target fresh. Returns None (no I/O) when nothing is
    running, so the worker emits nothing.
    """
    if not target.get("running"):
        return None
    from llama_launcher.services.metrics import (
        kv_ratio, decode_rate, counter_rate, prompt_progress,
    )
    port, host, key = target["port"], target["host"], target["key"]
    model_scope, poll = target["model_scope"], target["poll"]
    name, binary = target["name"], target["binary"]
    m = (metrics.fetch_metrics(port, model=model_scope, api_key=key, host=host)
         if target["metrics_on"] and poll else {})
    slots = (metrics.fetch_slots(port, model=model_scope, api_key=key, host=host)
             if poll else [])
    # Live generation tok/s from the n_decode_total counter delta -- the
    # predicted_tokens_seconds gauge only updates at request completion, so it
    # reads 0 during an in-flight generation. decode_now is handed back so the
    # controller can feed it in as decode_prev next tick.
    nd = m.get("llamacpp:n_decode_total")
    decode_now = (nd, time.monotonic()) if nd is not None else None
    gen_tok_s_live = decode_rate(target.get("decode_prev"), decode_now)
    # Live prompt tok/s the same way: the prompt_tokens_seconds gauge reads 0
    # mid-prefill; the processing slot's n_prompt_tokens_processed grows batch
    # by batch, so its delta is the live prefill rate.
    pp = prompt_progress(slots)
    prompt_now = (pp, time.monotonic()) if pp is not None else None
    prompt_tok_s_live = counter_rate(target.get("prompt_prev"), prompt_now)
    if target.get("kind") == "native" and target.get("pid"):
        st = native.proc_stats(target["pid"]) or {}
        uptime = ""     # native uptime is not tracked in v1
    else:
        mon_conn = target.get("mon_conn", "")
        st = runtime.stats(name, binary, connection=mon_conn) or {}
        uptime = _fmt_uptime(runtime.started_at(name, binary, connection=mon_conn))
    return {
        "tok_s": m.get("llamacpp:predicted_tokens_seconds"),
        "gen_tok_s_live": gen_tok_s_live,
        "decode_now": decode_now,
        "prompt_tok_s": m.get("llamacpp:prompt_tokens_seconds"),
        "prompt_tok_s_live": prompt_tok_s_live,
        "prompt_now": prompt_now,
        "kv_pct": kv_ratio(m, slots),
        "speculating": any(s.get("speculative") for s in slots),
        "gpus": gpu.query_gpus(target.get("gpu_ssh", "")),
        "cpu": st.get("cpu_perc", ""),
        "mem": st.get("mem_usage", ""),
        "uptime": uptime,
        "metrics_on": target["metrics_on"],
    }


def _instance_api_key_from(inst, by_name: dict, router_base_dir: str) -> str | None:
    """API key for polling one running instance, resolved from a profiles
    snapshot (never a fresh disk scan). A router reads its key store; a single
    server uses its stored --api-key. Mirrors the UI-thread _instance_api_key,
    but keyed off the snapshot so the whole table costs one list_profiles."""
    if inst.mode == "router":
        return api_key_store.read_api_key(router_base_dir, inst.profile)
    stored = by_name.get(inst.profile)
    return (stored.settings.get("api-key") or None) if stored else None


def _instance_summary_data(inst, by_name: dict, router_base_dir: str,
                           decode_prev: tuple | None = None) -> dict:
    """Per-row health + headline stat + structured tok_s/kv_pct for one instance
    (pure; blocking I/O). An embedding/rerank server has no tok/s (headline "ready");
    a generation server reports live n_decode_total-delta tok/s (falling back to
    the completion gauge when idle, so the card still shows the last run's rate)
    + KV% from /metrics + /slots. `decode_prev` is this row's previous
    (n_decode_total, monotonic) read; `decode_now` is handed back so the caller
    can feed it in next tick. Uses the profiles snapshot for key resolution
    instead of a per-row disk scan."""
    if not inst.running or inst.port is None:
        return {"health": "down", "stat": "", "tok_s": None, "kv_pct": None}
    hstatus = health.probe_health(inst.port, host=inst.host)
    if inst.embeddings or inst.reranking:
        stat = "ready" if hstatus == "ready" else ""
        return {"health": hstatus, "stat": stat, "tok_s": None, "kv_pct": None}
    key = _instance_api_key_from(inst, by_name, router_base_dir)
    m = metrics.fetch_metrics(inst.port, host=inst.host, api_key=key)
    slots = metrics.fetch_slots(inst.port, host=inst.host, api_key=key)
    nd = m.get("llamacpp:n_decode_total")
    decode_now = (nd, time.monotonic()) if nd is not None else None
    live = metrics.counter_rate(decode_prev, decode_now)
    tok = live if live is not None else m.get("llamacpp:predicted_tokens_seconds")
    kv = metrics.kv_ratio(m, slots)
    stat = f"{tok:.0f} tok/s" if tok else ("ready" if hstatus == "ready" else "")
    return {"health": hstatus, "stat": stat, "tok_s": tok, "kv_pct": kv,
            "decode_now": decode_now}


def build_instances_data(target: dict) -> dict:
    """Gather the instances table from a primitives-only `target`, off the UI thread.

    Does every blocking call the table needs -- the `list_launcher_containers`
    subprocess (once per enabled node) and the per-instance health/metrics
    probes -- plus a SINGLE `list_profiles` scan shared across all rows (the
    old synchronous refresh did one scan to build the list and one more per
    instance to resolve its key, so N servers cost N+1 scans on the UI thread
    every tick). Returns the built Instance list (for selection lookup) and
    the plain row dicts to render.

    `target["nodes"]` (when present) is a list of plain-dict node snapshots
    ({name, connection, host, binary, enabled}) -- see _instances_target. A
    disabled node is skipped; a node whose `list_launcher_containers` call
    raises OSError (unreachable) contributes zero rows instead of aborting
    the whole gather, so one dead remote can't blank the table for the rest.
    Callers that don't pass "nodes" (older targets, existing tests) get the
    prior local-only behaviour unchanged.
    """
    profiles = list_profiles(target["base_dir"])
    by_name = {p.name: p for p in profiles}
    nodes = target.get("nodes")
    legacy_target = nodes is None      # old single-node target with no "nodes" key
    if legacy_target:
        nodes = [{"name": "local", "connection": "", "host": "",
                  "binary": target["binary"], "enabled": True}]
    instances = []
    for nd in nodes:
        if not nd.get("enabled", True):
            continue
        binary = nd["binary"]
        node_name = nd.get("name", "local")
        conn = nd.get("connection", "")
        try:
            if legacy_target:
                # Preserve the exact old call shape (no `connection` kwarg) so
                # callers/tests built before nodes existed are unaffected.
                container_rows = runtime.list_launcher_containers(binary)
            else:
                container_rows = runtime.list_launcher_containers(binary, connection=conn)
        except OSError:
            container_rows = []
        native_rows = (native.list_native_instances(target["base_dir"])
                       if node_name == "local" else [])
        instances.extend(build_instances(
            container_rows + native_rows, profiles, binary,
            node=node_name, node_host=nd.get("host", "")))
    instances.sort(key=lambda i: (not i.running, i.node, i.name))
    decode_prev_by_key = target.get("decode_prev_by_key") or {}
    decode_now_by_key = {}
    rows = []
    for inst in instances:
        # An rpc-worker container shares its pool head's `llama-launcher.profile`
        # label (Task 4) so the pool joins as one profile -- which would
        # otherwise render the worker's card with the SAME title/port as the
        # head. Override the display-only fields with the worker's own
        # identity (StatCard.update_row would append `node` again if left as
        # the plain node name, so the already-worker_card_title-composed title
        # carries it and "local" suppresses the second append). An rpc-server
        # has no HTTP endpoint, so DON'T run the health/metrics probe: its
        # `inst.port` resolves to the pool head's port, which would make a
        # worker card show the HEAD's tok/s (misleading). Report up/down only.
        if inst.mode == "rpc-worker":
            summ = {"health": "ready" if inst.running else "down",
                    "stat": "", "tok_s": None, "kv_pct": None}
            profile_disp, port_disp, node_disp = worker_card_title(inst), None, "local"
        else:
            rate_key = f"{inst.node}/{inst.name}"
            summ = _instance_summary_data(inst, by_name, target["router_base_dir"],
                                          decode_prev=decode_prev_by_key.get(rate_key))
            if summ.get("decode_now") is not None:
                decode_now_by_key[rate_key] = summ["decode_now"]
            profile_disp, port_disp, node_disp = inst.profile, inst.port, inst.node
        rows.append({"name": inst.name, "profile": profile_disp, "port": port_disp,
                     "running": inst.running, "health": summ["health"],
                     "stat": summ["stat"], "tok_s": summ["tok_s"],
                     "kv_pct": summ["kv_pct"], "embeddings": inst.embeddings,
                     "reranking": inst.reranking, "mode": inst.mode, "node": node_disp})
    return {"instances": instances, "rows": rows,
            "decode_now_by_key": decode_now_by_key}


class _MonitorGather(QRunnable):
    """Run build_monitor_data() off the UI thread on the global thread pool.

    Delivery is by writing plain attributes on the owning MonitorController
    (assignment is atomic under the GIL) rather than a cross-thread Qt signal,
    and the task holds a reference to that controller so it (and transitively
    the window, via controller.window) can't be garbage-collected mid-gather.
    That sidesteps the "C++ object deleted while its thread runs" aborts a
    persistent QThread worker risks when a MainWindow is torn down (notably
    across tests).
    """
    def __init__(self, owner, target):
        super().__init__()
        self._owner = owner
        self._target = target

    def run(self):
        try:
            data = build_monitor_data(self._target)
        except Exception:            # noqa: BLE001 - worker must never raise
            data = None
        self._owner._monitor_result = data
        self._owner._monitor_inflight = False


class _InstancesGather(QRunnable):
    """Run build_instances_data() off the UI thread on the global thread pool.

    Same delivery model as _MonitorGather: it writes the result onto the owning
    MonitorController (GIL-atomic assignment) instead of emitting a cross-thread
    Qt signal, and holds a reference to the controller so it can't be
    garbage-collected mid-gather. A gather that raises leaves the previous result
    intact (rather than blanking the table) but always clears the in-flight flag.
    Note a `podman ps` failure is NOT a raise -- it returns [] -- so an empty
    result still overwrites; _render_instances guards the auto-clear against that.
    """
    def __init__(self, owner, target):
        super().__init__()
        self._owner = owner
        self._target = target

    def run(self):
        try:
            data = build_instances_data(self._target)
        except Exception:            # noqa: BLE001 - worker must never raise
            data = None
        if data is not None:
            self._owner._instances_result = data
        self._owner._instances_inflight = False


class StatsWorker(QThread):
    """Polls a snapshot builder off the UI thread and emits each result.

    The builder is injected (so it's testable without Qt); the worker owns only
    the loop + stop flag. Sleeps in small slices so stop() is responsive.
    """
    sampled = Signal(object)      # StatsSnapshot

    def __init__(self, builder, interval_ms: int = 1000, parent=None):
        super().__init__(parent)
        self._builder = builder
        self._interval_ms = interval_ms
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        while not self._stop:
            try:
                snap = self._builder()
            except Exception:
                snap = None
            if snap is not None and not self._stop:
                self.sampled.emit(snap)
            slept = 0
            while slept < self._interval_ms and not self._stop:
                self.msleep(50)
                slept += 50


class MonitorController:
    """Status/instances/monitor/log-follower/stats/router-poll behavior.

    Owns the state a live poll needs between ticks (the last gather result,
    the active-instance override, the log follower/flush timer, the stats
    worker, and the router/props/spec-decode caches) and the methods that
    read/write it. `window` is the MainWindow this controller is attached to
    -- widgets stay on the window (it builds them); this controller only owns
    behavior + the plain state above.
    """

    def __init__(self, window):
        self.window = window

        # -- router / props / spec-decode caches (cleared on profile load) --
        self._router_statuses: dict = {}
        self._spec_prev = None      # previous /metrics spec-decode counter read
        self._decode_prev = None    # previous (n_decode_total, monotonic) for live tok/s
        self._prompt_prev = None    # previous (prompt_progress, monotonic) for live prefill tok/s
        self._cards_decode_prev = {}   # per-card "node/name" -> (n_decode_total, monotonic)
        self._props = None          # cached /props for the current model load
        self._props_model = None    # router-polled model id the cache is keyed on

        # -- instances / active-instance override --
        self._instances = []              # last-built Instance list (for selection lookup)
        self._active_instance = None      # Instance being monitored, or None -> current profile

        # -- the blocking Monitor summary (podman stats, nvidia-smi, /metrics,
        # /slots) is gathered off the UI thread by a short-lived _MonitorGather
        # task on the global thread pool (dispatched from update_status);
        # update_status only snapshots the cheap primitives into _monitor_target
        # and renders the latest result. No persistent worker thread -- a slow
        # podman stats can't stutter the GUI, and there's no long-lived QThread
        # to abort on teardown.
        self._monitor_target = {"running": False}
        self._monitor_result = None
        self._monitor_inflight = False

        # -- instances table: same off-UI-thread pattern as the monitor summary.
        # The blocking list_launcher_containers subprocess + the N per-instance
        # health/metrics probes are gathered by a pooled _InstancesGather task;
        # update_status renders the last result and dispatches a fresh gather.
        self._instances_result = None
        self._instances_inflight = False

        # -- log follower --
        self._log_proc = None
        # Follower output is buffered here and drained to the widget by a timer
        # at a bounded rate. A per-chunk widget append floods the UI thread when
        # a model logs heavily during generation (the "unusable while generating"
        # freeze); coalescing bursts into one append per tick keeps the event
        # loop free for user input.
        self._log_pending: list[str] = []
        self._log_flush_timer = QTimer(self.window)
        self._log_flush_timer.setInterval(100)     # 10 Hz -> ~1 widget write / 100 ms
        self._log_flush_timer.timeout.connect(self._flush_log)

        # -- stats dock worker --
        self._stats_worker = None
        self._cpu_sampler = None
        # (container_name, binary) snapshot the StatsWorker reads instead of
        # touching GUI widgets from its own thread -- see _refresh_stats_target.
        self._stats_target = ("", "podman")

    # -- teardown ------------------------------------------------------------
    def drain(self) -> None:
        self._stop_stats_worker()
        # Let any in-flight monitor gather finish (bounded) so it isn't writing
        # to the window during teardown; the pool's threads outlive the window,
        # so there's nothing to abort even if this times out.
        QThreadPool.globalInstance().waitForDone(3000)

    # -- stats dock ------------------------------------------------------------
    def _on_stats_visibility(self, visible: bool) -> None:
        # Keep the toolbar button in sync when the dock is closed via its own X,
        # and persist the state.
        if self.window.stats_toggle_btn.isChecked() != visible:
            self.window.stats_toggle_btn.setChecked(visible)
        if visible:
            self._start_stats_worker()
        else:
            self._stop_stats_worker()
        self._save_stats_config()

    def _refresh_stats_target(self) -> None:
        # Read GUI/profile state on the UI thread only; the worker reads the
        # resulting plain tuple (safe under the GIL), never the widgets.
        self._stats_target = (self._monitored_container_name(),
                              self.window._configure_panel.current_profile().runtime.binary)

    def _start_stats_worker(self) -> None:
        if self._stats_worker is not None and self._stats_worker.isRunning():
            return
        from llama_launcher.services import stats as stats_svc
        from llama_launcher.services.sysstat import CpuSampler
        self._cpu_sampler = CpuSampler()
        self._refresh_stats_target()

        def _build():
            name, binary = self._stats_target
            return stats_svc.build_snapshot(name, binary, self._cpu_sampler)

        self._stats_worker = StatsWorker(_build, interval_ms=1000, parent=self.window)
        self._stats_worker.sampled.connect(self.window.stats_panel.update_stats)
        self._stats_worker.start()

    def _stop_stats_worker(self) -> None:
        w = self._stats_worker
        if w is None:
            return
        w.stop()
        from PySide6.QtCore import QCoreApplication
        for _ in range(100):            # ~2s ceiling, pump events between waits
            if w.wait(20):
                break
            QCoreApplication.processEvents()
        else:
            w.terminate()
            w.wait(100)
        # Release the stopped worker so it doesn't linger as a child of the
        # window for its whole lifetime; each dock open builds a fresh one.
        w.deleteLater()
        self._stats_worker = None

    def _save_stats_config(self) -> None:
        from llama_launcher.ui.main_window import base_dir
        cfg = load_config(base_dir())
        cfg["stats_open"] = self.window.stats_dock.isVisibleTo(self.window)
        cfg["stats_width"] = self.window.stats_dock.width() or cfg.get("stats_width", 320)
        save_config(cfg, base_dir())

    # -- spec-decode / props / router polling --------------------------------
    def _update_spec_stats(self, p: Profile) -> None:
        """Derive spec-decode acceptance from /metrics counters.

        The log-scraped acceptance line stays the source when --metrics is off,
        and is the only source in server mode. In router mode the log belongs to
        the ROUTER, so a child model's acceptance can only be attributed via
        ?model=<id> -- which is what these counters are for.
        """
        if not p.settings.get("metrics"):
            return
        model_scope = self._router_pollable_model() if p.mode == "router" else None
        if p.mode == "router" and model_scope is None:
            return
        text = metrics.fetch_metrics_text(
            p.settings.get("port", 8080), model=model_scope,
            api_key=self._poll_api_key(p),
            host=dial_host(p.runtime.bind_host))
        cur = spec_counters(text) if text else None
        if cur is None:
            return
        if self._spec_prev is not None:
            self.window.monitor_panel.set_draft_stats(spec_delta(self._spec_prev, cur),
                                                       source="counters")
        self._spec_prev = cur

    def _poll_api_key(self, p: Profile) -> str | None:
        """API key for authenticating Monitor polls -- the key the running server
        actually uses. A router reads it from --api-key-file (our key store); a
        single server uses its own --api-key setting. Returns None when there's
        no key (so no Authorization header is sent). Without this, a single
        server started with --api-key rejected /props, /metrics and /slots polls
        with "Invalid API Key" (only /health, which needs no key, still worked).
        """
        if p.mode == "router":
            return api_key_store.read_api_key(self.window.router_base_dir(), p.name)
        return p.settings.get("api-key") or None

    def _refresh_props(self, p: Profile) -> str | None:
        """Fetch /props once per model load and cache it (static per load).

        Keyed on the router-polled model id so a router swap re-fetches; in
        single-model mode the key is constant, so it fetches exactly once.

        Returns the router model key it resolved (None in server mode, or
        when router mode has nothing loaded), so callers that already need
        that value -- e.g. the benchmark-availability gate -- can reuse it
        instead of polling `_router_pollable_model()` again.
        """
        port = p.settings.get("port", 8080)
        key = self._poll_api_key(p)
        if p.mode == "router":
            host = self._router_host(p)
            model_key = self._router_pollable_model()
        else:
            host = dial_host(p.runtime.bind_host)
            model_key = None
        if self._props is not None and self._props_model == model_key:
            return model_key
        info = metrics.fetch_props(port, api_key=key, host=host)
        if info is None:
            return model_key           # leave cache empty; retry next ready poll
        self._props = info
        self._props_model = model_key
        self.window.monitor_panel.set_props(info)
        return model_key

    def _router_host(self, p: Profile) -> str:
        """The address the GUI itself dials for this profile."""
        return dial_host(p.runtime.bind_host)

    def refresh_router_models(self) -> None:
        p = self.window._configure_panel.current_profile()
        if p.mode != "router":
            return
        host = self._router_host(p)
        port = p.settings.get("port", 8080)
        key = api_key_store.read_api_key(self.window.router_base_dir(), p.name)
        models = router_api.list_models(host, port, key)
        if models is None:            # unreachable, as opposed to serving nothing
            self._router_statuses = {}
            self.window.router_models_table.set_models([])
            self.window._set_router_connected(False)
            return
        self._router_statuses = {m.id: m.status for m in models}
        self.window.router_models_table.set_models(models)
        self.window._set_router_connected(True)

    def _router_pollable_model(self) -> str | None:
        """The resident model to scope Monitor polling to, or None.

        Sleeping and unloaded models are skipped: there is nothing to measure,
        and not polling them is the whole point of an idle-unloading host."""
        for model_id, status in self._router_statuses.items():
            if status == "loaded":
                return model_id
        return None

    def _on_router_load(self, model_id: str) -> None:
        p = self.window._configure_panel.current_profile()
        key = api_key_store.read_api_key(self.window.router_base_dir(), p.name)
        ok = router_api.load_model(self._router_host(p), p.settings.get("port", 8080),
                                   key, model_id)
        self.refresh_router_models()
        if not ok:
            # Silently discarding this left a failed load looking identical to a
            # slow one: the row just stayed "unloaded" forever.
            self.window._set_router_error(f"load failed: {model_id}")

    def _on_router_unload(self, model_id: str) -> None:
        p = self.window._configure_panel.current_profile()
        key = api_key_store.read_api_key(self.window.router_base_dir(), p.name)
        ok = router_api.unload_model(self._router_host(p), p.settings.get("port", 8080),
                                     key, model_id)
        self.refresh_router_models()
        if not ok:
            self.window._set_router_error(f"unload failed: {model_id}")

    # -- status poll -----------------------------------------------------------
    def update_status(self):
        # Keep the (container_name, binary) snapshot StatsWorker reads current
        # while the dock is open and the user switches profile/instance --
        # this UI-thread timer callback is the only writer of _stats_target.
        if self._stats_worker is not None and self._stats_worker.isRunning():
            self._refresh_stats_target()
        p = self._monitored_profile()
        if not runtime.binary_available(p.runtime.binary):
            self.window.status_label.setText("● stopped")
            self.window._configure_panel.web_ui_btn.setEnabled(False)
            self.window.benchmark_panel.set_benchmark_available(False)
            self._monitor_target = {"running": False}
            self._monitor_result = None
            self._decode_prev = None
            self._prompt_prev = None
            return
        name = self._monitored_container_name()
        state = runtime.container_state(name, p.runtime.binary,
                                         connection=self._connection_for_node(p.runtime.node))
        # Default the gather target to "don't poll"; the running branch below
        # overwrites it with the live snapshot. Nothing is gathered off-thread
        # until update_status confirms the container is up. (_monitor_result is
        # left intact here so the running branch can render the last gather;
        # it's cleared below only when nothing is running.)
        self._monitor_target = {"running": False}
        hstatus = health.probe_health(p.settings.get("port", 8080),
                                     host=dial_host(p.runtime.bind_host)) \
            if state == "running" else "down"
        self.window.status_label.setText("● " + health.derive_status(state, hstatus))
        self.window._configure_panel.web_ui_btn.setEnabled(state == "running")
        router_model_key = None
        if state == "running":
            if not self._log_follower_active():
                self._start_log_follower()
            if hstatus == "ready":
                router_model_key = self._refresh_props(p)
            # Snapshot the poll inputs (cheap) and hand the blocking gather to a
            # pooled task off the UI thread. router_model_key was already
            # resolved by _refresh_props above -- reuse it so a router tick polls
            # _router_pollable_model() once. Render the previous tick's result
            # (up to one tick stale) and dispatch a fresh gather unless one is
            # already in flight (so a slow podman stats can't pile up).
            # Render the previous gather, then advance the decode baseline to
            # ITS reading before snapshotting the next target -- so the gather
            # dispatched below measures the rate over the gap to that reading.
            if self._monitor_result is not None:
                self.window.monitor_panel.update_stats(self._monitor_result)
                self._decode_prev = self._monitor_result.get("decode_now")
                self._prompt_prev = self._monitor_result.get("prompt_now")
            self._monitor_target = self._compute_monitor_target(
                running=True, model_scope=router_model_key)
            if not self._monitor_inflight:
                self._monitor_inflight = True
                QThreadPool.globalInstance().start(
                    _MonitorGather(self, self._monitor_target))
            self._update_spec_stats(p)
        else:
            # Nothing running: drop the last gather so a stale summary isn't
            # rendered on the next start before a fresh gather completes, and
            # drop the decode baseline so the first rate after the next start
            # isn't measured across the downtime.
            self._monitor_result = None
            self._decode_prev = None
            self._prompt_prev = None
        if p.mode == "router":
            if state == "running":
                self.refresh_router_models()
            else:
                # Router stopped/removed: clear the stale model list + connected
                # state so a dead router doesn't keep showing load/unload rows.
                self._router_statuses = {}
                self.window.router_models_table.set_models([])
                self.window._set_router_connected(False)
        # router_model_key was resolved from _refresh_props above (when ready)
        # rather than polled again here, so this reuses that single call to
        # _router_pollable_model() instead of doubling it.
        ready = state == "running" and hstatus == "ready"
        if ready and p.mode == "router":
            ready = router_model_key is not None
        self.window.benchmark_panel.set_benchmark_available(ready)
        self._refresh_instances_list()

    def _refresh_instances_list(self) -> None:
        # Render the previous gather (up to one tick stale) then dispatch a fresh
        # one off the UI thread, guarded so a slow gather can't pile up. Mirrors
        # the monitor-summary cadence -- the podman ps subprocess and the N
        # per-instance probes must not run on the event loop.
        self._render_instances()
        if not self._instances_inflight:
            self._instances_inflight = True
            QThreadPool.globalInstance().start(_InstancesGather(self, self._instances_target()))

    def _instances_target(self) -> dict:
        """Snapshot the primitives the gather needs (UI thread only), including
        every registered node as plain dicts -- no live Node objects, since the
        gather runs off the UI thread and must not touch shared state."""
        from llama_launcher.ui.main_window import base_dir
        return {"binary": self.window._configure_panel.current_profile().runtime.binary,
                "base_dir": base_dir(),
                "router_base_dir": self.window.router_base_dir(),
                "decode_prev_by_key": dict(self._cards_decode_prev),
                "nodes": [
                    {"name": n.name, "connection": connection_for(n), "host": host_of(n),
                     "binary": n.binary, "enabled": n.enabled}
                    for n in load_nodes(base_dir())
                ]}

    def _render_instances(self) -> None:
        result = self._instances_result
        if result is None:
            return
        self._instances = result["instances"]
        # Advance the per-card decode baselines to THIS result's readings so the
        # next gather measures each card's live rate over the gap to them
        # (mirrors the focused monitor's decode_prev handling). Idempotent when
        # the same result is re-rendered while a gather is in flight.
        self._cards_decode_prev = result.get("decode_now_by_key", {})
        # Auto-clear a monitored instance whose container has dropped out of the
        # fresh list (crash / external stop) so the Monitor falls back to the form
        # profile and retargets the log follower instead of stranding on a dead
        # target -- previously only explicit Stop/Remove cleared _active_instance.
        # Gate on a NON-EMPTY list: `podman ps` failing (daemon hiccup, timeout)
        # returns [] rather than raising, so an empty list is ambiguous and must
        # not be read as "my instance vanished" -- only a populated list that
        # omits the instance is genuine evidence it was stopped externally.
        if (self._active_instance is not None and self._instances
                and self._active_instance.name not in {i.name for i in self._instances}):
            self._active_instance = None
            self._start_log_follower()
        self.window.monitor_panel.set_instance_cards(
            {"rows": result["rows"], "selected_name": self._monitored_container_name()})

    def _on_instance_selected(self, name: str) -> None:
        inst = next((i for i in self._instances if i.name == name), None)
        # Selecting the form's own container means "monitor the current profile" (fallback).
        self._active_instance = None if (inst is None or name == self.window._container_name()) else inst
        self._start_log_follower()          # retarget the follower at the new container
        self.update_status()

    def _rpc_pool_profile_for(self, inst) -> Profile | None:
        """The stored Profile for `inst` (or the current form profile when inst
        is None) when it is the HEAD of an rpc pool -- i.e. its runtime.launch_mode
        is "rpc" -- else None. An rpc-worker's own card must never be routed
        through the pool stop (it stops just that one worker container), so it
        short-circuits to None regardless of what its (shared, head's) profile
        says."""
        if inst is not None and inst.mode == "rpc-worker":
            return None
        name = inst.profile if inst is not None \
            else self.window._configure_panel.current_profile().name
        prof = next((p for p in list_profiles(self.window.base_dir()) if p.name == name), None)
        return prof if prof is not None and prof.runtime.launch_mode == "rpc" else None

    def _on_instance_stop(self, name: str) -> None:
        import signal
        inst = next((i for i in self._instances if i.name == name), None)
        if inst is not None and inst.kind == "native":
            if inst.pid is not None:
                native.stop_native(inst.pid, signal.SIGTERM)
                _schedule_sigkill(inst.pid, inst.stop_timeout)
            self.update_status()
            if self._active_instance is not None and self._active_instance.name == name:
                self._active_instance = None
            return
        pool_profile = self._rpc_pool_profile_for(inst)
        if pool_profile is not None:
            # Coordinated stop: the head AND every worker container, over each
            # worker's own node -- a bare `podman stop` on just the head would
            # leave the workers running. Dispatch OFF the UI thread (per-worker
            # `podman stop` + ssh tunnel teardown would otherwise freeze the
            # GUI), reusing LaunchController's pool seam; it refreshes status on
            # completion via _on_pool_stopped.
            self.window._launch.stop_pool_async(pool_profile)
        else:
            binary = inst.binary if inst is not None \
                else self.window._configure_panel.current_profile().runtime.binary
            timeout = inst.stop_timeout if inst is not None else DEFAULT_STOP_TIMEOUT
            node_name = inst.node if inst is not None \
                else self.window._configure_panel.current_profile().runtime.node
            connection = self._connection_for_node(node_name)
            self.window._launch._spawn_async(
                runtime.stop_argv(name, binary, timeout=timeout, connection=connection),
                on_done=self.update_status)
        if self._active_instance is not None and self._active_instance.name == name:
            self._active_instance = None

    def _on_instance_remove(self, name: str) -> None:
        # A stopped launcher container lingers in `podman ps -a` with no useful
        # action; remove it so the instances list can be cleared.
        from llama_launcher.ui.main_window import base_dir
        inst = next((i for i in self._instances if i.name == name), None)
        if inst is not None and inst.kind == "native":
            native.remove_native(name, base_dir())
            self.update_status()
        else:
            binary = inst.binary if inst is not None \
                else self.window._configure_panel.current_profile().runtime.binary
            node_name = inst.node if inst is not None \
                else self.window._configure_panel.current_profile().runtime.node
            connection = self._connection_for_node(node_name)
            self.window._launch._spawn_async(
                runtime.rm_argv(name, binary, connection=connection), on_done=self.update_status)
        if self._active_instance is not None and self._active_instance.name == name:
            self._active_instance = None

    def _monitored_profile(self) -> Profile:
        from llama_launcher.ui.main_window import base_dir
        inst = self._active_instance
        if inst is None:
            return self.window._configure_panel.current_profile()
        stored = next((p for p in list_profiles(base_dir())
                       if p.name == inst.profile), None)
        # Trust the running container's real mode (from its label) over the
        # stored profile: a profile saved as a single server but launched as a
        # router (or one never saved) would otherwise be polled in server mode,
        # sending no router key -> "Invalid API Key". When stored is gone or its
        # mode disagrees with what's actually running, synthesize identity from
        # the instance so polls use the right mode/key/host/port.
        if stored is not None and stored.mode == inst.mode:
            return stored
        return Profile(
            name=inst.profile, mode=inst.mode,
            runtime=Runtime(bind_host=inst.host, node=inst.node),
            settings={"port": inst.port or 8080,
                      "embeddings": inst.embeddings, "reranking": inst.reranking,
                      "metrics": bool(stored.settings.get("metrics")) if stored else False},
        )

    def _monitored_container_name(self) -> str:
        return self._active_instance.name if self._active_instance else self.window._container_name()

    def _connection_for_node(self, node_name: str) -> str:
        """The podman --connection name for a node ('' for local/missing)."""
        node = get_node(self.window.base_dir(), node_name)
        return connection_for(node) if node else ""

    def instance_summary(self, inst) -> dict:
        """Per-row health + headline stat for one instance. Thin UI-side wrapper
        over the pure _instance_summary_data (which the off-thread gather also
        uses); resolves the profiles snapshot + router key dir on demand. Returns
        {"health", "stat"} -- the old dead "running" key (rows read inst.running
        directly) was dropped."""
        from llama_launcher.ui.main_window import base_dir
        by_name = {p.name: p for p in list_profiles(base_dir())}
        return _instance_summary_data(inst, by_name, self.window.router_base_dir())

    def _compute_monitor_target(self, running: bool, model_scope=None) -> dict:
        """Snapshot the poll inputs into a primitives-only dict on the UI thread.

        The monitor worker reads this (never the widgets/profile) and calls
        build_monitor_data() off-thread, so the blocking gather -- podman stats,
        nvidia-smi, /metrics, /slots -- no longer runs on the UI thread. Only
        the cheap derivation of these primitives stays here.

        The router model to scope polling to is passed in (already resolved by
        _refresh_props this tick) rather than re-polled here, so a router tick
        calls _router_pollable_model() exactly once.
        """
        if not running:
            return {"running": False}
        p = self._monitored_profile()
        node = get_node(self.window.base_dir(), p.runtime.node)
        host, key, ms, poll = (dial_host(p.runtime.bind_host),
                               self._poll_api_key(p), None, True)
        if p.mode == "router":
            host = self._router_host(p)
            ms = model_scope
            poll = ms is not None
        if node is not None and node.kind == "remote":
            # A remote profile's bind_host is typically 0.0.0.0 -> dial_host()
            # resolves that to 127.0.0.1, which would poll the LOCAL host's
            # loopback instead of the node the server actually runs on.
            host = host_of(node)
        # Never send the api key to an arbitrary host: the bearer token goes only
        # to loopback or the profile's registered remote node. A profile whose
        # bind_host is some other address (validation blocks saving one, but a
        # loaded-not-launched profile is still polled) gets an unauthenticated
        # poll rather than leaking the secret.
        allowed_key_host = (host in ("127.0.0.1", "localhost", "::1", "[::1]")
                            or (node is not None and node.kind == "remote"
                                and host == host_of(node)))
        if not allowed_key_host:
            key = None
        return {
            "running": True,
            "port": p.settings.get("port", 8080),
            "metrics_on": bool(p.settings.get("metrics")),
            "host": host, "key": key, "model_scope": ms, "poll": poll,
            "name": self._monitored_container_name(),
            "binary": p.runtime.binary,
            "kind": self._active_instance.kind if self._active_instance is not None else "container",
            "pid": self._active_instance.pid if self._active_instance is not None else None,
            "gpu_ssh": focused_gpu_ssh(p.runtime.node, self.window.base_dir()),
            "mon_conn": connection_for(node) if node else "",
            "decode_prev": self._decode_prev,
            "prompt_prev": self._prompt_prev,
        }

    def collect_monitor_data(self) -> dict:
        """Gather the Monitor summary synchronously (used by tests and any
        direct caller). The live poll instead snapshots _compute_monitor_target()
        and lets the worker call build_monitor_data() off the UI thread."""
        p = self._monitored_profile()
        ms = self._router_pollable_model() if p.mode == "router" else None
        return build_monitor_data(
            self._compute_monitor_target(running=True, model_scope=ms)) or {}

    # -- log follower ------------------------------------------------------
    def _log_follower_active(self) -> bool:
        from PySide6.QtCore import QProcess
        return (self._log_proc is not None
                and self._log_proc.state() != QProcess.NotRunning)

    def _start_log_follower(self):
        from PySide6.QtCore import QProcess
        self._stop_log_follower()
        p = self._monitored_profile()
        name = self._monitored_container_name()
        active = self._active_instance
        if active is not None and active.kind == "native":
            from llama_launcher.ui.main_window import base_dir
            logpath = native.native_log_path(base_dir(), active.profile)
            if not logpath.exists():
                return
            argv = native.logs_argv(str(logpath))
        else:
            # Attaching `podman logs -f` before the container exists just prints
            # "no such container" and exits, stranding the logs pane on that error.
            # Skip until it exists; update_status() retries once it's running and
            # `podman logs` replays from the beginning, so no early output is lost.
            connection = self._connection_for_node(p.runtime.node)
            if not runtime.container_exists(name, p.runtime.binary, connection=connection):
                return
            argv = runtime.logs_argv(name, p.runtime.binary, connection=connection)
        proc = QProcess(self.window)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(
            lambda: self._enqueue_log(bytes(proc.readAllStandardOutput()).decode("utf-8", "replace")))
        proc.start(argv[0], argv[1:])
        self._log_proc = proc

    def _enqueue_log(self, text: str) -> None:
        """Buffer a chunk of follower output and ensure the flush timer runs.

        Called from the follower's readyRead, so it must be cheap: it only
        appends to the buffer. The expensive widget write happens in _flush_log
        at 10 Hz, so a flood of readyRead during generation can't starve the UI
        thread of user-input events.
        """
        self._log_pending.append(text)
        if not self._log_flush_timer.isActive():
            self._log_flush_timer.start()

    def _flush_log(self) -> None:
        """Drain all buffered follower output to the panel in a single append.

        Stops the timer once the buffer is empty (an idle server shouldn't run a
        10 Hz timer forever); the next _enqueue_log restarts it.
        """
        if not self._log_pending:
            self._log_flush_timer.stop()
            return
        text = "".join(self._log_pending)
        self._log_pending.clear()
        self.window.monitor_panel.append_log(text)

    def _stop_log_follower(self):
        # Stop the flush timer and drop any buffered tail so switching/stopping a
        # container can't leak one container's trailing lines into the next.
        self._log_flush_timer.stop()
        self._log_pending.clear()
        if self._log_proc is not None:
            self._log_proc.kill()
            self._log_proc = None
