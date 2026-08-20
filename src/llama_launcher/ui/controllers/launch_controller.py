from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import QMessageBox, QInputDialog

from llama_launcher.core.command_builder import build_command
from llama_launcher.core import vram
from llama_launcher.core.nodes import connection_for
from llama_launcher.services import runtime, terminal, registry, model_info, gpu, native, rpc
from llama_launcher.services import benchmark_store
from llama_launcher.services.registry import split_image, variant_prefix
from llama_launcher.services import api_key as api_key_store
from llama_launcher.store.profiles import default_base_dir, load_config
from llama_launcher.store.nodes import get_node


class _UpdateWorker(QThread):
    found = Signal(str)
    failed = Signal(str)

    def __init__(self, repo: str, prefix: str, parent=None):
        super().__init__(parent)
        self._repo = repo
        self._prefix = prefix

    def run(self):
        try:
            tag = registry.fetch_latest(self._repo, self._prefix)
            if tag:
                self.found.emit(tag)
        except Exception as e:            # noqa: BLE001 - surfaced to the user
            self.failed.emit(str(e))


class LaunchController:
    """Owns launch/stop/restart/enable-metrics + image fetch/detect/update behavior.

    Widgets stay on the window (built by ConfigurePanel); this controller only
    owns behavior plus the plain state below (_fetch_worker/_update_worker/
    _stop_proc/_update_timer). Members this controller itself owns (e.g.
    `_spawn_async`, `_validate_or_warn`, `on_launch`) are called directly as
    `self.<method>(...)`; widgets and methods owned by other panels/controllers
    go through `self.window._<owner>.<x>` (e.g. `self.window._configure_panel.
    image_edit`, `self.window._monitor.update_status`). Test-suite patches
    that target this controller's own methods (e.g. `_spawn_async`) now patch
    `LaunchController` directly, not `MainWindow`.
    """

    def __init__(self, window):
        self.window = window

        self._stop_proc = None
        self._fetch_worker = None
        self._update_worker = None

        # The update-check timer. A singleShot fired 3s after construction so
        # the window (image field etc.) is fully built by the time it runs --
        # only created when the user hasn't opted out via config.
        self._update_timer = None
        from llama_launcher.ui.main_window import base_dir
        if load_config(base_dir()).get("update_check", True):
            self._update_timer = QTimer(self.window)
            self._update_timer.setSingleShot(True)
            self._update_timer.timeout.connect(self.run_update_check)
            self._update_timer.start(3000)

    # -- teardown -------------------------------------------------------------
    def drain(self) -> None:
        """Drain any in-flight registry-fetch / update-check QThread so closing
        the window mid-fetch can't destroy a running QThread (abort/crash).
        _UpdateWorker.run() is a blocking network call with no cancel flag, so
        wait with a ceiling and terminate() as a last-resort backstop.
        """
        if self._update_timer is not None:
            self._update_timer.stop()

        from PySide6.QtCore import QCoreApplication
        for _attr in ("_fetch_worker", "_update_worker"):
            w = getattr(self, _attr, None)
            if w is None or not w.isRunning():
                continue
            for _ in range(100):            # ~2s ceiling
                if w.wait(20):
                    break
                QCoreApplication.processEvents()
            else:
                w.terminate()
                w.wait(100)

    # -- launch / stop / restart ----------------------------------------------
    def _connection_for_profile(self, profile) -> str:
        """The podman --connection name for this profile's node ('' = local)."""
        node = get_node(self.window.base_dir(), profile.runtime.node)
        return connection_for(node) if node else ""

    def on_launch(self):
        if not self._validate_or_warn():
            return
        p = self.window._configure_panel.current_profile()
        connection = self._connection_for_profile(p)

        if p.runtime.launch_mode == "native":
            # Refuse a relaunch over an already-running native instance for
            # this profile: launch_native would spawn a second llama-server
            # that fails to bind the in-use port and exits, while write_entry
            # overwrites the registry with that dead PID -- orphaning the
            # original process (still holding the port/VRAM) with no
            # registry entry, invisible and unstoppable from the UI.
            live = native.list_native_instances(default_base_dir())
            name = native.native_name(p.name)
            if any(row.get("name") == name or row.get("profile") == p.name
                   for row in live):
                self._report_launch_error(
                    f"A native server for profile '{p.name}' is already "
                    f"running. Stop it first.", show_dialog=True)
                return

            self.window.monitor_panel.reset()
            self.window.benchmark_panel.reset()
            self.window.monitor_panel.set_endpoints(
                p.settings.get("port", 8080),
                bool(p.settings.get("embeddings")),
                bool(p.settings.get("reranking")))
            from datetime import datetime
            res = native.launch_native(p, default_base_dir(),
                                       now_iso=datetime.now().isoformat())
            if not res.ok:
                self._report_launch_error(res.error, show_dialog=True)
            else:
                self.window._monitor.update_status()
            return

        if p.runtime.launch_mode == "rpc":
            # Refuse a relaunch over an already-running pool: rpc.launch_pool
            # starts by tearing down the CURRENT pool's live ssh tunnels
            # (so a stale one isn't orphaned on relaunch), then the worker
            # `run` collides on the still-live `llama-<slug>-rpc0` container
            # name and fails -- leaving a healthy pool degraded with a
            # confusing error instead of a clean refusal.
            if runtime.container_state(self.window._container_name(),
                                       p.runtime.binary, connection="") == "running":
                self._report_launch_error(
                    f"An RPC pool for profile '{p.name}' is already "
                    f"running. Stop it before relaunching.", show_dialog=True)
                return
            self._launch_pool(p)
            return

        if p.mode == "router":
            router_host_dir, warnings = self.window.prepare_router_files()
            if warnings:
                QMessageBox.warning(self.window, "Preset warnings", "\n".join(warnings))
            argv = build_command(p, router_host_dir=router_host_dir, connection=connection)
            # Relaunching over a LIVE router would drop a resident model and any
            # in-flight harness requests, so confirm before tearing it down.
            if runtime.container_state(self.window._container_name(),
                                       p.runtime.binary, connection=connection) == "running":
                answer = QMessageBox.question(
                    self.window, "Router already running",
                    "This router is already running. Relaunching stops it, "
                    "unloading any resident model and dropping in-flight "
                    "requests. Continue?")
                if answer != QMessageBox.Yes:
                    return

            # A stopped container of the same name would block the new run,
            # since router mode deliberately omits --rm. Chain rather than fire
            # both at once: _spawn_async is asynchronous, so an unchained run
            # would race the removal and lose with "name already in use".
            self.window.monitor_panel.reset()
            self.window.benchmark_panel.reset()
            self.window.benchmark_panel.set_benchmark_history(
                benchmark_store.load(default_base_dir(), p.name))
            self.window._monitor._spec_prev = None
            self.window._monitor._props = None
            self.window._monitor._props_model = None
            self._spawn_async(
                runtime.rm_argv(self.window._container_name(), p.runtime.binary,
                                connection=connection),
                on_done=lambda: self._spawn_async(
                    argv, on_done=self.window._monitor.update_status,
                    # Detached means no terminal, so a bad image ref or a CDI
                    # failure would otherwise produce nothing but a status label
                    # stuck on "stopped".
                    on_error=self._report_launch_error))
            self.window.refresh_router_panel_header()
            return

        warn = self.vram_check()
        if warn:
            QMessageBox.warning(self.window, "VRAM check", warn)
        self.window.monitor_panel.reset()
        self.window.benchmark_panel.reset()
        self.window.benchmark_panel.set_benchmark_history(
            benchmark_store.load(default_base_dir(), p.name))
        self.window.monitor_panel.set_endpoints(
            p.settings.get("port", 8080),
            bool(p.settings.get("embeddings")),
            bool(p.settings.get("reranking")),
        )
        if p.runtime.detached:
            argv = build_command(p, detach=True, connection=connection)
            # Detached drops --rm, so a stale stopped container of this name
            # would block the run with "name already in use". Remove it first,
            # then chain the run (mirrors the router branch above). on_error
            # surfaces bad image / CDI / flag failures the terminal used to
            # show -- show_dialog is fixed here, at launch time, so a later
            # profile/mode switch before the error fires can't change it.
            self._spawn_async(
                runtime.rm_argv(self.window._container_name(), p.runtime.binary,
                                connection=connection),
                on_done=lambda: self._spawn_async(
                    argv, on_done=self.window._monitor.update_status,
                    on_error=lambda e=None: self._report_launch_error(
                        e, show_dialog=True)))
        else:
            argv = build_command(p, connection=connection)
            # A `terminal` config value overrides detection; otherwise auto-detect
            # an installed terminal (konsole on KDE, ptyxis/gnome-terminal on GNOME,
            # ...). A missing terminal raises instead of crashing the launch.
            template = load_config(default_base_dir()).get("terminal")
            try:
                terminal.launch(argv, template=template)
            except terminal.NoTerminalError as exc:
                self._report_launch_error(str(exc), show_dialog=True)
        # Don't attach the log follower here: the container is created
        # asynchronously and doesn't exist yet. update_status() starts the
        # follower once it is actually running (podman logs replays from the
        # start, so no early output is missed).

    def _launch_pool(self, p) -> None:
        """RPC pool launch: worker(s) + head are started synchronously by the
        pool orchestrator (each worker's `run -d` returns fast), so there's no
        async spawn/callback plumbing here -- just report the outcome."""
        res = rpc.launch_pool(p, self.window.base_dir())
        if res.ok:
            self.window._monitor.update_status()
        else:
            self._report_launch_error(res.error, show_dialog=True)

    def _spawn_async(self, argv: list[str], on_done=None, on_error=None):
        """Run argv in the background via QProcess so the UI thread never blocks.
        Calls on_done() when the process finishes.

        on_error(text) receives podman's stderr on a non-zero exit, and a
        message if the binary could not be started at all — `finished` is NOT
        emitted on FailedToStart, so without errorOccurred a launch that never
        began would report nothing whatsoever.
        """
        from PySide6.QtCore import QProcess
        proc = QProcess(self.window)
        if on_done is not None:
            proc.finished.connect(lambda *_: on_done())
        if on_error is not None:
            def _finished(code, _status):
                if code != 0:
                    err = bytes(proc.readAllStandardError()).decode(errors="replace").strip()
                    on_error(err or f"{argv[0]} exited {code}")
            proc.finished.connect(_finished)
            proc.errorOccurred.connect(
                lambda _e: on_error(f"could not run {argv[0]}: {proc.errorString()}"))
        proc.start(argv[0], argv[1:])
        return proc

    def on_stop(self):
        p = self.window._configure_panel.current_profile()
        if p.runtime.launch_mode == "rpc":
            rpc.stop_pool(p, self.window.base_dir())
            self.window._monitor.update_status()
            return

        # Stop the log follower immediately; run `podman stop` asynchronously so a
        # slow stop (podman waits up to its grace period for SIGTERM) never freezes
        # the GUI — which previously made Stop look like it did nothing.
        self.window._monitor._stop_log_follower()
        connection = self._connection_for_profile(p)
        self.window.status_label.setText("● stopping…")
        argv = runtime.stop_argv(self.window._container_name(), p.runtime.binary,
                                 timeout=p.runtime.stop_timeout, connection=connection)
        self._stop_proc = self._spawn_async(argv, on_done=self.window._monitor.update_status)

    def on_restart(self):
        self.window._monitor._stop_log_follower()
        p = self.window._configure_panel.current_profile()
        connection = self._connection_for_profile(p)
        self.window.status_label.setText("● restarting…")
        argv = runtime.stop_argv(self.window._container_name(), p.runtime.binary,
                                 timeout=p.runtime.stop_timeout, connection=connection)
        # Launch only after the stop completes, so the new container's --name/port
        # don't collide with the one being torn down.
        self._stop_proc = self._spawn_async(argv, on_done=self.on_launch)

    def _on_enable_metrics(self):
        self.window._configure_panel._widgets["metrics"].set_value(True)
        self.on_restart()

    # -- validation / vram / error reporting ----------------------------------
    def _validate_or_warn(self) -> bool:
        p = self.window._configure_panel.current_profile()
        # The key must exist before the exposure rule is evaluated: a router
        # always gets one at launch, but this runs before prepare_router_files.
        if p.mode == "router":
            api_key_store.prepare_launch_key(self.window.router_base_dir(), p)
        issues = self.window._configure_panel.router_issues()
        errors = [i for i in issues if i.level == "error"]
        if errors:
            QMessageBox.critical(self.window, "Cannot launch",
                                 "\n".join(i.message for i in errors))
            return False
        warns = [i for i in issues if i.level == "warning"]
        if warns:
            QMessageBox.warning(self.window, "Warnings", "\n".join(i.message for i in warns))
        return True

    def vram_check(self) -> str | None:
        p = self.window._configure_panel.current_profile()
        meta, weights, _caps = model_info.inspect_model(p.model, self.window._configure_panel.mounts_panel.mounts()) if p.model else (None, None, None)
        gpus = gpu.query_gpus()
        if meta is None or not gpus or not meta.n_layers or not meta.n_embd:
            return None
        mib = 1024 * 1024
        free_per_gpu = [g.mem_free_mib * mib for g in gpus]
        # Budget depends on how the model is placed: split across all GPUs (the
        # default) means their combined free VRAM; split-mode none means one card.
        split_mode = p.settings.get("split-mode", "layer")
        main_gpu = p.settings.get("main-gpu", 0)
        free = vram.available_free_bytes(free_per_gpu, split_mode, main_gpu)
        ctx = p.settings.get("ctx-size") or meta.ctx_train or 4096
        est = vram.estimate(
            n_layers=meta.n_layers, n_head=meta.n_head or 1,
            n_head_kv=meta.n_head_kv or meta.n_head or 1, n_embd=meta.n_embd, ctx=ctx,
            k_quant=p.settings.get("cache-type-k", "f16"),
            v_quant=p.settings.get("cache-type-v", "f16"),
            weights_bytes=weights or 0,
        )
        ok, margin = vram.fits(est.total_bytes, free)
        if ok:
            return None
        gib = 1024 ** 3
        # Show the per-GPU breakdown when the budget spans multiple cards, so the
        # "free" number is transparent (e.g. "14.7 + 7.3 = 22.0 GiB across 2 GPUs").
        if len(free_per_gpu) > 1 and split_mode != "none":
            parts = " + ".join(f"{b/gib:.1f}" for b in free_per_gpu)
            free_txt = f"~{free/gib:.1f} GiB ({parts} across {len(free_per_gpu)} GPUs)"
        else:
            free_txt = f"~{free/gib:.1f} GiB"
        return (f"Estimated VRAM need ~{est.total_bytes/gib:.1f} GiB exceeds free "
                f"{free_txt} by ~{-margin/gib:.1f} GiB. It may not fit — "
                f"consider quantized KV cache (-ctk/-ctv q8_0) or a higher --n-cpu-moe. "
                f"(Estimate is conservative; --n-cpu-moe/-ngl reduce actual GPU use.)")

    def _report_launch_error(self, text: str = None, *, show_dialog: bool = False) -> None:
        """Show why a detached launch -- router or server -- failed to start.
        Routed to the status banners (non-modal: this fires from a QProcess
        signal, which tests drive); a detached SERVER launch also pops a
        QMessageBox, since a Monitor-tab-only user may never see the
        Configure tab's banner and would otherwise miss the failure entirely.

        `show_dialog` is decided by the CALLER at launch time (when the
        profile's mode is known synchronously), not re-derived here from
        live UI state: this fires from an async QProcess callback, possibly
        seconds later, by which point the user may have switched profiles
        or flipped the mode combo -- current_profile() at that moment would
        no longer describe the launch that actually failed."""
        self.window.status_label.setText("● failed to start")
        reason = (f"launch failed: {text.splitlines()[-1][:200]}"
                  if text else "launch failed")
        self.window._set_router_error(reason)
        if show_dialog:
            QMessageBox.critical(self.window, "Launch failed", reason)

    def adopt_running_containers(self) -> list:
        """Containers this launcher owns, so a detached router survives a GUI restart."""
        p = self.window._configure_panel.current_profile()
        return runtime.list_launcher_containers(p.runtime.binary)

    # -- image fetch / detect / update-check -----------------------------------
    def on_fetch_latest(self):
        repo, tag = split_image(self.window._configure_panel.image_edit.text())
        if not repo:
            QMessageBox.information(
                self.window, "No image",
                "Set or Detect an image first — Fetch latest looks up the newest "
                "build for the image's repository.")
            return
        prefix = variant_prefix(tag) if tag else "server-cuda12"
        self._fetch_repo = repo
        self._fetch_got_result = False
        self.window._configure_panel.fetch_btn.setEnabled(False)
        self.window._configure_panel.fetch_btn.setText("Fetching…")
        self.window._configure_panel.update_badge.setEnabled(False)
        worker = _UpdateWorker(repo, prefix, parent=self.window)
        worker.found.connect(self._on_fetch_found)
        worker.failed.connect(self._on_fetch_failed)
        worker.finished.connect(self._on_fetch_finished)   # QThread built-in
        self._fetch_worker = worker
        worker.start()

    def _on_fetch_found(self, tag: str) -> None:
        self._fetch_got_result = True
        image = f"{self._fetch_repo}:{tag}"
        self.window._configure_panel.image_edit.setText(image)
        QMessageBox.information(
            self.window, "Latest build",
            f"Image set to {image}.\n\nThis only updates the tag — the build is NOT "
            f"downloaded. Pull it with:\n  podman pull {image}\n(or docker pull).")

    def _on_fetch_failed(self, msg: str) -> None:
        self._fetch_got_result = True
        QMessageBox.warning(
            self.window, "Fetch failed", f"Couldn't fetch the latest build:\n{msg}")

    def _on_fetch_finished(self) -> None:
        self.window._configure_panel.fetch_btn.setEnabled(True)
        self.window._configure_panel.fetch_btn.setText("Fetch latest")
        self.window._configure_panel.update_badge.setEnabled(True)
        if not self._fetch_got_result:
            QMessageBox.information(
                self.window, "Latest build", "No newer build found for this image.")

    def detect_image(self):
        binary = self.window._configure_panel.binary_combo.currentText()
        engine = self.window._configure_panel.engine_combo.currentData() or "llama.cpp"
        images = runtime.list_local_images(binary, engine)
        if not images:
            example = ("ghcr.io/ikawrakow/ik-llama-cpp:cu12-server"
                       if engine == "ik_llama.cpp"
                       else "ghcr.io/ggml-org/llama.cpp:server")
            QMessageBox.information(
                self.window, "Detect image",
                f"No local {engine} images found for '{binary}'.\n"
                f"Pull one (e.g. {binary} pull {example}) or type the image yourself.")
            return
        if len(images) == 1:
            self.window._configure_panel.image_edit.setText(images[0])
            return
        choice, ok = QInputDialog.getItem(
            self.window, "Detect image", f"Local {engine} images:", images, 0, False)
        if ok and choice:
            self.window._configure_panel.image_edit.setText(choice)

    def _autofill_image_if_empty(self):
        if self.window._configure_panel.image_edit.text().strip():
            return
        images = runtime.list_local_images(
            self.window._configure_panel.binary_combo.currentText(),
            self.window._configure_panel.engine_combo.currentData() or "llama.cpp")
        if len(images) == 1:
            self.window._configure_panel.image_edit.setText(images[0])

    def check_for_update(self, tags: list[str]) -> str | None:
        repo, tag = split_image(self.window._configure_panel.image_edit.text())
        if not tag:
            return None
        prefix = variant_prefix(tag)
        latest = registry.latest_build_tag(tags, prefix)
        if latest and latest != tag:
            return latest
        return None

    def run_update_check(self):
        repo, tag = split_image(self.window._configure_panel.image_edit.text())
        if not repo or not tag:
            return
        prefix = variant_prefix(tag)
        current_tag = tag

        def _on_found(latest: str):
            if latest != current_tag:
                m = registry._BUILD_RE.match(latest)
                build_id = f"b{m.group('num')}" if m else latest
                self.window._configure_panel.update_badge.setText(f"newer build {build_id} available")
                self.window._configure_panel.update_badge.setVisible(True)

        worker = _UpdateWorker(repo, prefix, parent=self.window)
        worker.found.connect(_on_found)
        self._update_worker = worker
        worker.start()
