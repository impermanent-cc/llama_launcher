import dataclasses
import os
import subprocess
from pathlib import Path

import datetime

from PySide6.QtCore import Qt, QObject, QRunnable, QThread, QThreadPool, Signal
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit,
    QCheckBox, QGroupBox, QScrollArea, QLabel, QPlainTextEdit, QPushButton,
    QMessageBox, QFileDialog, QInputDialog, QTabWidget, QDockWidget
)

from llama_launcher.core.spec import (
    Profile, Mount, Runtime, RouterMember, member_model_id, slugify,
)
from llama_launcher.core.router_preset import render_preset
from llama_launcher.core.settings_catalog import CATALOG, member_catalog, router_catalog
from llama_launcher.core.command_builder import build_command
from llama_launcher.core.pathmap import host_to_container
from llama_launcher.core.validation import (
    validate, Issue, LOOPBACK_HOSTS, dial_host,
)
from llama_launcher.store.profiles import (
    default_base_dir, list_profiles, save_profile, delete_profile,
    load_config, save_config, profile_to_dict, resolve_member_pairs,
)
from llama_launcher.core.instances import Instance, build_instances
from llama_launcher.services import runtime, terminal, registry, health, metrics, gpu, model_info
from llama_launcher.services import benchmark, benchmark_store
from llama_launcher.core import vram
from llama_launcher.core.mtp_stats import spec_counters, spec_delta
from llama_launcher.core import report as report_mod
from llama_launcher.services.registry import split_image, variant_prefix
from llama_launcher.ui.dialogs.report_dialog import ReportDialog
import posixpath
from llama_launcher.core.capabilities import describe_relevance, Tier, suggestions as compute_suggestions
from llama_launcher.core.pathmap import container_to_host
from llama_launcher.ui.widgets.setting_widgets import make_widget, SuggestionDot
from llama_launcher.ui.widgets.no_wheel import NoWheelComboBox
from llama_launcher.ui.panels.mounts_panel import MountsPanel
from llama_launcher.ui.panels.lora_panel import LoraPanel
from llama_launcher.ui.widgets.collapsible import CollapsibleSection
from llama_launcher.ui.panels.monitor_panel import MonitorPanel
from llama_launcher.ui.panels.benchmark_panel import BenchmarkPanel
from llama_launcher.ui.panels.stats_panel import StatsPanel
from llama_launcher.ui.widgets.api_key_box import ApiKeyBox
from llama_launcher.ui.widgets.harness_info_box import HarnessInfoBox
from llama_launcher.ui.widgets.router_models_table import RouterModelsTable
from llama_launcher.ui.widgets.status_banner import StatusBanner
from llama_launcher.services import api_key as api_key_store
from llama_launcher.services import router_api


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
    from llama_launcher.services.metrics import kv_ratio
    port, host, key = target["port"], target["host"], target["key"]
    model_scope, poll = target["model_scope"], target["poll"]
    name, binary = target["name"], target["binary"]
    m = (metrics.fetch_metrics(port, model=model_scope, api_key=key, host=host)
         if target["metrics_on"] and poll else {})
    slots = (metrics.fetch_slots(port, model=model_scope, api_key=key, host=host)
             if poll else [])
    st = runtime.stats(name, binary) or {}
    uptime = _fmt_uptime(runtime.started_at(name, binary))
    return {
        "tok_s": m.get("llamacpp:predicted_tokens_seconds"),
        "prompt_tok_s": m.get("llamacpp:prompt_tokens_seconds"),
        "kv_pct": kv_ratio(m, slots),
        "speculating": any(s.get("speculative") for s in slots),
        "gpus": gpu.query_gpus(),
        "cpu": st.get("cpu_perc", ""),
        "mem": st.get("mem_usage", ""),
        "uptime": uptime,
        "metrics_on": target["metrics_on"],
    }


class _MonitorGather(QRunnable):
    """Run build_monitor_data() off the UI thread on the global thread pool.

    Delivery is by writing plain attributes on the window (assignment is atomic
    under the GIL) rather than a cross-thread Qt signal, and the task holds a
    reference to the window so it can't be garbage-collected mid-gather. That
    sidesteps the "C++ object deleted while its thread runs" aborts a persistent
    QThread worker risks when a MainWindow is torn down (notably across tests).
    """
    def __init__(self, window, target):
        super().__init__()
        self._window = window
        self._target = target

    def run(self):
        try:
            data = build_monitor_data(self._target)
        except Exception:            # noqa: BLE001 - worker must never raise
            data = None
        self._window._monitor_result = data
        self._window._monitor_inflight = False


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


def base_dir():
    return default_base_dir()


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Llama Launcher")
        self._profile = Profile(name="New Profile")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Configure tab = the existing left+right body
        configure_tab = QWidget()
        body = QHBoxLayout(configure_tab)
        # Trim the body's own margins so the Environment/Settings columns get a
        # little extra vertical room (the bottom command-preview strip is kept
        # compact rather than made draggable).
        body.setContentsMargins(4, 4, 4, 2)

        # LEFT: environment (image + model only for v1 binding; mounts editor TODO-UI)
        left = QGroupBox("Environment")
        left_form = QFormLayout(left)
        self._left_form = left_form
        self.image_edit = QLineEdit()
        self.model_edit = QLineEdit()
        self.binary_combo = NoWheelComboBox(); self.binary_combo.addItems(["podman", "docker"])
        self.gpu_combo = NoWheelComboBox()
        for _gpu_label, _gpu_val in (
            ("CDI — --device nvidia.com/gpu=all (recommended)", "cdi"),
            ("Legacy — --gpus all", "gpus-all"),
            ("None — no GPU passthrough", "none"),
        ):
            self.gpu_combo.addItem(_gpu_label, _gpu_val)
        for w in (self.image_edit, self.model_edit):
            w.textChanged.connect(self.refresh_preview)
        self.binary_combo.currentTextChanged.connect(self.refresh_preview)
        self.gpu_combo.currentTextChanged.connect(self.refresh_preview)
        self.update_badge = QPushButton("")
        self.update_badge.setFlat(True)
        self.update_badge.setVisible(False)
        self.update_badge.clicked.connect(self.on_fetch_latest)
        self.detect_image_btn = QPushButton("Detect")
        self.detect_image_btn.setToolTip(
            "Fill the Image field from llama.cpp images already pulled locally "
            "(podman/docker images).")
        self.detect_image_btn.clicked.connect(self.detect_image)
        self.fetch_btn = QPushButton("Fetch latest")
        self.fetch_btn.setToolTip(
            "Query the container registry (GHCR) for the newest build tag matching "
            "this image's repo and variant, and update the Image field. Requires an "
            "image to be set (use Detect or type one). This updates the tag only — it "
            "does NOT download the build; pull it with podman/docker pull.")
        self.fetch_btn.clicked.connect(self.on_fetch_latest)
        image_row = QHBoxLayout()
        image_row.setContentsMargins(0, 0, 0, 0)
        image_row.addWidget(self.image_edit, 1)
        image_row.addWidget(self.detect_image_btn)
        image_row.addWidget(self.fetch_btn)
        image_row.addWidget(self.update_badge)
        image_widget = QWidget()
        image_widget.setLayout(image_row)
        from PySide6.QtWidgets import QTableWidget, QHeaderView

        # NoWheel: an unfocused scroll over these must not silently flip launch
        # semantics (terminal vs detached) or expose the port to the network.
        self.mode_combo = NoWheelComboBox()
        self.mode_combo.addItem("Single server", "server")
        self.mode_combo.addItem("Router (headless host)", "router")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self.bind_host_combo = NoWheelComboBox()
        self.bind_host_combo.setEditable(True)
        self.bind_host_combo.addItems(["127.0.0.1", "0.0.0.0"])
        self.bind_host_combo.currentTextChanged.connect(self.refresh_preview)

        # A table, not a list: model id / load-on-startup / stop-timeout are all
        # per-member settings the spec requires to be editable, and inline
        # editing keeps them reachable without a modal dialog (which would hang
        # the headless test run).
        self.members_list = QTableWidget(0, 4)
        self.members_list.setHorizontalHeaderLabels(
            ["Profile", "Model id (harness)", "Load at start", "Stop timeout (s)"])
        for _col, _tip in enumerate((
            "A saved model profile to serve. Configure its GPU layers / MoE / "
            "context in that profile (single-server mode), then add it here.",
            "Name the harness calls this model. Empty = derived from the profile name.",
            "Load this member as soon as the router starts (otherwise on first request).",
            "Seconds to wait after unload before force-killing this member's container.",
        )):
            item = self.members_list.horizontalHeaderItem(_col)
            if item is not None:
                item.setToolTip(_tip)
        self.members_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.members_list.verticalHeader().setVisible(False)
        self.members_list.setMaximumHeight(140)
        self.members_list.itemChanged.connect(lambda _i: self.refresh_preview())
        self.add_member_btn = QPushButton("Add member…")
        self.add_member_btn.clicked.connect(self._on_add_member)
        self.remove_member_btn = QPushButton("Remove member")
        self.remove_member_btn.clicked.connect(self._on_remove_member)
        self.edit_member_btn = QPushButton("Edit member…")
        self.edit_member_btn.setToolTip(
            "Load the selected member's profile into the form to set its GPU "
            "layers, MoE offload, context, etc. (double-clicking a row does the same).")
        self.edit_member_btn.clicked.connect(self._on_edit_member)
        members_row = QHBoxLayout()
        members_row.setContentsMargins(0, 0, 0, 0)
        members_row.addWidget(self.add_member_btn)
        members_row.addWidget(self.edit_member_btn)
        members_row.addWidget(self.remove_member_btn)
        members_widget = QWidget()
        self._members_row = members_widget
        members_box = QVBoxLayout(members_widget)
        members_box.setContentsMargins(0, 0, 0, 0)
        members_box.addWidget(self.members_list)
        members_box.addLayout(members_row)
        self.members_guidance = QLabel(
            "Each member is a saved model profile — set its GPU layers, MoE offload, "
            "and context in that profile (single-server mode), then add it here.")
        self.members_guidance.setWordWrap(True)
        self.members_guidance.setStyleSheet("QLabel { color: palette(mid); }")
        members_box.addWidget(self.members_guidance)
        self.members_list.cellDoubleClicked.connect(
            lambda _r, _c: self._on_edit_member() if _c == 0 else None)

        left_form.addRow("Mode", self.mode_combo)
        left_form.addRow("Bind address", self.bind_host_combo)
        left_form.addRow("Router members", members_widget)

        left_form.addRow("Image", image_widget)
        left_form.addRow("Model", self._field_with_browse(self.model_edit))
        left_form.addRow("Runtime", self.binary_combo)
        left_form.addRow("GPU", self.gpu_combo)

        # mounts editor
        self.mounts_panel = MountsPanel()
        self.mounts_panel.changed.connect(self.refresh_preview)
        left_form.addRow("Folders", self.mounts_panel)
        self.mmproj_edit = QLineEdit(); self.mmproj_edit.textChanged.connect(self.refresh_preview)
        self.draft_model_edit = QLineEdit(); self.draft_model_edit.textChanged.connect(self.refresh_preview)
        self.raw_edit = QLineEdit(); self.raw_edit.textChanged.connect(self.refresh_preview)
        self._mmproj_dot = SuggestionDot(self)
        self._draft_model_dot = SuggestionDot(self)
        left_form.addRow("mmproj", self._field_with_browse(self.mmproj_edit, self._mmproj_dot))
        left_form.addRow("draft model", self._field_with_browse(self.draft_model_edit, self._draft_model_dot))
        self.lora_panel = LoraPanel()
        self.lora_panel.changed.connect(self.refresh_preview)
        self.lora_section = CollapsibleSection("LoRA adapters", self.lora_panel, collapsed=True)
        left_form.addRow(self.lora_section)
        left_form.addRow("Raw args", self.raw_edit)
        self.extra_args_edit = QLineEdit()
        self.extra_args_edit.textChanged.connect(self.refresh_preview)
        left_form.addRow("Extra podman args", self.extra_args_edit)
        self.selinux_check = QCheckBox("Disable SELinux labels (--security-opt=label=disable)")
        self.selinux_check.toggled.connect(self.refresh_preview)
        left_form.addRow(self.selinux_check)

        # Set tooltips on Environment field widgets
        self.mode_combo.setToolTip(
            "Single server runs one model. Router (headless host) serves several "
            "member models on one port with an API key, loading them on demand.")
        self.bind_host_combo.setToolTip(
            "Address the server binds to. 127.0.0.1 = this machine only; "
            "0.0.0.0 = reachable from the network (an API key is then required).")
        self.image_edit.setToolTip(
            "Container image for llama-server, e.g. "
            "ghcr.io/ggml-org/llama.cpp:server-cuda. Use Detect to list local images.")
        self.model_edit.setToolTip(
            "Path to the .gguf model as seen INSIDE the container, e.g. "
            "/models/Qwen3-A3B-Q4.gguf (add a Folder that maps the host dir).")
        self.binary_combo.setToolTip("Container runtime used to launch the server.")
        self.gpu_combo.setToolTip(
            "GPU passthrough. CDI (nvidia.com/gpu=all) is recommended on modern "
            "NVIDIA + podman; Legacy uses --gpus all; None runs CPU-only.")
        self.mmproj_edit.setToolTip(
            "Optional multimodal projector .gguf (container path) for vision models.")
        self.draft_model_edit.setToolTip(
            "Optional small draft model .gguf (container path) for speculative decoding.")
        self.raw_edit.setToolTip(
            "Extra llama-server flags appended verbatim, e.g. --temp 0.6 --top-k 20. "
            "Duplicates of structured settings are de-duplicated.")
        self.extra_args_edit.setToolTip(
            "Extra flags for the container runtime (podman/docker) itself, e.g. "
            "--shm-size=1g. Not passed to llama-server.")
        self.selinux_check.setToolTip(
            "Add --security-opt=label=disable. Needed on some SELinux hosts when a "
            "mounted model dir is otherwise unreadable in the container.")
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left)
        body.addWidget(left_scroll, 3)

        # RIGHT: settings grouped, scrollable
        self._widgets: dict[str, object] = {}
        right_scroll = QScrollArea(); right_scroll.setWidgetResizable(True)
        right_inner = QWidget(); right_layout = QVBoxLayout(right_inner)
        groups: dict[str, QFormLayout] = {}
        self._group_boxes: dict[str, QGroupBox] = {}
        # key -> (form layout, widget), so _on_mode_changed can hide whole rows
        # (label included) for settings that don't apply to the current mode.
        self._setting_rows: dict[str, tuple] = {}
        for key, setting in CATALOG.items():
            if setting.group not in groups:
                box = QGroupBox(setting.group)
                groups[setting.group] = QFormLayout(box)
                self._group_boxes[setting.group] = box
                right_layout.addWidget(box)
            w = make_widget(setting)
            w.changed.connect(self.refresh_preview)
            self._widgets[key] = w
            groups[setting.group].addRow(setting.flag, w)
            self._setting_rows[key] = (groups[setting.group], w)
        # When load-mode is active it supersedes --no-mmap/--mlock (see
        # command_builder), so gray those legacy checkboxes out to signal they're
        # ignored. Kept in sync on edit, mode change and profile load.
        if "load-mode" in self._widgets:
            self._widgets["load-mode"].changed.connect(self._sync_load_mode_legacy)
        right_scroll.setWidget(right_inner)
        body.addWidget(right_scroll, 2)
        self._last_caps = None
        self._router_statuses: dict = {}
        self._spec_prev = None      # previous /metrics spec-decode counter read
        self._props = None          # cached /props for the current model load
        self._props_model = None    # router-polled model id the cache is keyed on
        self._benchmark_thread = None
        self._benchmark_worker = None
        self._benchmark_profile_name = None
        self.configure_tab = configure_tab

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(configure_tab, "Configure")
        self.monitor_panel = MonitorPanel()
        self.monitor_panel.enable_metrics_requested.connect(self._on_enable_metrics)
        self.monitor_panel.instance_selected.connect(self._on_instance_selected)
        self.monitor_panel.instance_stop_requested.connect(self._on_instance_stop)
        self.monitor_panel.instance_remove_requested.connect(self._on_instance_remove)
        # Scroll the Monitor tab (like Configure): a short window otherwise
        # squeezes the log to a few lines. The log now owns the tab (benchmark
        # moved to its own tab), so it fills the height.
        monitor_scroll = QScrollArea()
        monitor_scroll.setWidgetResizable(True)
        monitor_scroll.setWidget(self.monitor_panel)
        self.tabs.addTab(monitor_scroll, "Monitor")

        self.api_key_box = ApiKeyBox()
        self.api_key_box.key_scope_changed.connect(self._on_key_scope_changed)
        self.api_key_box.key_saved.connect(self._on_key_saved)
        self.harness_box = HarnessInfoBox()
        self.configure_status = StatusBanner()
        self.router_models_table = RouterModelsTable()
        self.router_models_table.load_requested.connect(self._on_router_load)
        self.router_models_table.unload_requested.connect(self._on_router_unload)
        self.monitor_status = StatusBanner()
        self.monitor_panel.add_status_banner(self.monitor_status)
        self.monitor_panel.add_below_log(self.router_models_table)

        self.benchmark_panel = BenchmarkPanel()
        self.benchmark_panel.benchmark_run_requested.connect(self._on_benchmark_run)
        self.benchmark_panel.benchmark_cancel_requested.connect(self._on_benchmark_cancel)
        self.benchmark_panel.benchmark_clear_requested.connect(self._on_benchmark_clear)
        self.tabs.addTab(self.benchmark_panel, "Benchmark")
        self._configure_tab = configure_tab
        self.tabs.currentChanged.connect(self._on_tab_changed)
        # Stretch=1 so the tab body (Configure's Environment/Settings columns)
        # fills the vertical space down to the command-preview strip, rather
        # than leaving dead space where the router-only widgets are hidden.
        root.addWidget(self.tabs, 1)

        # Right-hand stats dock (GPU / system / container). Hidden by default;
        # a checkable button in the top bar toggles it. Polling is wired in the
        # StatsWorker task and runs only while the dock is visible.
        self.stats_panel = StatsPanel()
        self.stats_dock = QDockWidget("Stats", self)
        self.stats_dock.setObjectName("stats_dock")
        self.stats_dock.setWidget(self.stats_panel)
        self.stats_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, self.stats_dock)
        self.stats_dock.hide()
        # Initialised here (ahead of the other worker handles, set further
        # below near _fetch_worker/_update_worker) because the visibility/
        # toggle signals wired immediately below -- and the stats_open
        # restore later in __init__, which programmatically checks the
        # toggle button -- can synchronously reach _start_stats_worker()
        # before __init__ reaches that later block.
        self._stats_worker = None
        self._cpu_sampler = None
        # (container_name, binary) snapshot the StatsWorker reads instead of
        # touching GUI widgets from its own thread -- see _refresh_stats_target.
        self._stats_target = ("", "podman")
        # Same early-init hazard as above: _refresh_stats_target() (called
        # from _start_stats_worker) reads _monitored_container_name(), which
        # reads _active_instance -- so it must exist before the stats_open
        # restore/toggle wiring below can reach it. (Moved up from its
        # original spot further down, near _fetch_worker/_update_worker.)
        self._active_instance = None      # Instance being monitored, or None -> current profile
        self.stats_dock.visibilityChanged.connect(self._on_stats_visibility)

        # BOTTOM: command preview is config-only; wrap it in one container so
        # it can be hidden on the Monitor/Router/Benchmark tabs (see
        # _on_tab_changed). Launch/Stop/etc stay shared below.
        self._config_bottom = QWidget()
        config_bottom_box = QVBoxLayout(self._config_bottom)
        config_bottom_box.setContentsMargins(0, 0, 0, 0)
        config_bottom_box.setSpacing(3)   # tighten the bottom strip's dead space
        self.model_meta_label = QLabel("")
        config_bottom_box.addWidget(self.model_meta_label)
        self.model_edit.textChanged.connect(lambda _: self.apply_model_caps())
        self.mounts_panel.changed.connect(self.apply_model_caps)
        config_bottom_box.addWidget(self.configure_status)
        config_bottom_box.addWidget(self.api_key_box)
        config_bottom_box.addWidget(self.harness_box)
        preview_row = QHBoxLayout()
        self.preview = QPlainTextEdit(); self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(90)
        preview_row.addWidget(self.preview, 1)
        self.export_sh_btn = QPushButton("Export .sh")
        self.export_sh_btn.clicked.connect(self._on_export_sh)
        config_bottom_box.addWidget(QLabel("Command preview:"))
        config_bottom_box.addLayout(preview_row)
        root.addWidget(self._config_bottom)
        buttons = QHBoxLayout()
        self.launch_btn = QPushButton("▶ Launch")
        self.stop_btn = QPushButton("■ Stop")
        self.restart_btn = QPushButton("⟳ Restart")
        self.web_ui_btn = QPushButton("Open Web UI")
        self.web_ui_btn.setEnabled(False)
        self.web_ui_btn.clicked.connect(self.open_web_ui)
        for b in (self.launch_btn, self.stop_btn, self.restart_btn, self.web_ui_btn):
            buttons.addWidget(b)
        self.detached_check = QCheckBox("Run detached (no terminal window)")
        self.detached_check.setToolTip(
            "Launch without a terminal window; watch output on the Monitor "
            "tab and use Stop to shut it down.")
        self.detached_check.toggled.connect(self.refresh_preview)
        buttons.addWidget(self.detached_check)
        root.addLayout(buttons)

        # profile bar (added to the top of root via insertLayout)
        bar = QHBoxLayout()
        self.name_edit = QLineEdit(self._profile.name)
        self.name_edit.setPlaceholderText("Profile name")
        self.name_edit.setToolTip(
            "Name for this profile. Also sets the container name "
            "(llama-<name>), so pick something filesystem-friendly.")
        self.name_edit.textChanged.connect(self.refresh_preview)
        self.profile_combo = NoWheelComboBox()
        self.save_btn = QPushButton("Save")
        self.save_as_btn = QPushButton("Save As")
        self.delete_btn = QPushButton("Delete")
        self.report_btn = QPushButton("Generate report")
        self.stats_toggle_btn = QPushButton("📊 Stats")
        self.stats_toggle_btn.setCheckable(True)
        self.stats_toggle_btn.setToolTip("Show/hide the live stats panel (Ctrl+Shift+S)")
        self.status_label = QLabel("● stopped")
        bar.addWidget(QLabel("Name"))
        bar.addWidget(self.name_edit, 1)
        bar.addWidget(self.profile_combo, 1)
        for b in (self.save_btn, self.save_as_btn, self.delete_btn, self.report_btn,
                  self.stats_toggle_btn):
            bar.addWidget(b)
        bar.addWidget(self.status_label)
        root.insertLayout(0, bar)
        self.save_btn.clicked.connect(self.save_current_profile)
        self.save_as_btn.clicked.connect(self.save_as_profile)
        self.delete_btn.clicked.connect(self.delete_current_profile)
        self.report_btn.clicked.connect(self.on_generate_report)
        self.stats_toggle_btn.toggled.connect(self.stats_dock.setVisible)
        # Also drive worker start/stop straight off the toggle: QDockWidget's
        # visibilityChanged only fires once the top-level window has actually
        # been shown (isVisible() requires the whole ancestor chain mapped),
        # which never happens for an unshown/offscreen MainWindow -- e.g. in
        # tests, or if stats are toggled before the first show(). toggled()
        # fires synchronously regardless. _on_stats_visibility's start/stop
        # calls are idempotent, so double-firing (this + visibilityChanged,
        # once the window is shown) is harmless.
        self.stats_toggle_btn.toggled.connect(self._on_stats_visibility)
        from PySide6.QtGui import QShortcut, QKeySequence
        QShortcut(QKeySequence("Ctrl+Shift+S"), self,
                  activated=lambda: self.stats_toggle_btn.toggle())
        self.profile_combo.activated.connect(self._on_pick_profile)

        # lifecycle buttons
        self.launch_btn.clicked.connect(self.on_launch)
        self.stop_btn.clicked.connect(self.on_stop)
        self.restart_btn.clicked.connect(self.on_restart)

        self._reload_profile_list()

        # Apply the initial mode's visibility. The combo defaults to "server"
        # and startup loads no profile, so currentIndexChanged never fires --
        # without this, the router-only member widgets (created visible) show on
        # the default server-mode form until the user flips the mode combo.
        self._on_mode_changed()

        self.refresh_preview()

        _stats_cfg = load_config(base_dir())
        if _stats_cfg.get("stats_open", False):
            self.stats_toggle_btn.setChecked(True)     # shows the dock
        _w = int(_stats_cfg.get("stats_width", 320) or 320)
        self.resizeDocks([self.stats_dock], [_w], Qt.Horizontal)

        self._log_proc = None
        # Follower output is buffered here and drained to the widget by a timer
        # at a bounded rate. A per-chunk widget append floods the UI thread when
        # a model logs heavily during generation (the "unusable while generating"
        # freeze); coalescing bursts into one append per tick keeps the event
        # loop free for user input.
        self._log_pending: list[str] = []
        from PySide6.QtCore import QTimer
        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setInterval(100)     # 10 Hz -> ~1 widget write / 100 ms
        self._log_flush_timer.timeout.connect(self._flush_log)
        self._stop_proc = None
        self._fetch_worker = None
        self._update_worker = None
        self._instances = []              # last-built Instance list (for selection lookup)

        from PySide6.QtWidgets import QSystemTrayIcon, QMenu, QStyle
        from llama_launcher.ui.icon import app_icon
        self._really_quit = False
        # Apply our own SVG icon to the window. app.main() also sets it at the
        # application level, but doing it here means the icon is present even
        # when the window is constructed without going through main() (tests,
        # embedding), and is what the window-manager titlebar/Alt-Tab uses.
        self.setWindowIcon(app_icon())
        # Minimize-to-tray is OPT-IN: enabled only when the user turns it on in
        # config AND the desktop actually provides a system tray. By default,
        # closing the window quits the app (see closeEvent), so a tray-less or
        # half-wired session never strands the process behind a blocked terminal.
        self._minimize_to_tray = (
            bool(load_config(base_dir()).get("minimize_to_tray", False))
            and QSystemTrayIcon.isSystemTrayAvailable()
        )
        if self._minimize_to_tray:
            self.tray = QSystemTrayIcon(self)
            tray_icon = app_icon()
            if tray_icon.isNull():   # no asset/theme icon found — keep a visible fallback
                tray_icon = self.style().standardIcon(QStyle.SP_ComputerIcon)
            self.tray.setIcon(tray_icon)
            self.tray.setToolTip("Llama Launcher")
            menu = QMenu()
            menu.addAction("Show", self.showNormal)
            menu.addAction("Launch", self.on_launch)
            menu.addAction("Stop", self.on_stop)
            menu.addSeparator()
            menu.addAction("Quit", self.quit_app)
            self.tray.setContextMenu(menu)
            self.tray.show()
        else:
            self.tray = None

        self.lora_panel.set_browse_resolver(
            lambda h: host_to_container(h, self.mounts_panel.mounts())
        )

        # Auto-insert the local image when there's exactly one and none is set yet.
        self._autofill_image_if_empty()

        # The blocking Monitor summary (podman stats, nvidia-smi, /metrics,
        # /slots) is gathered off the UI thread by a short-lived _MonitorGather
        # task on the global thread pool (dispatched from update_status);
        # update_status only snapshots the cheap primitives into _monitor_target
        # and renders the latest result. No persistent worker thread -- a slow
        # podman stats can't stutter the GUI, and there's no long-lived QThread
        # to abort on teardown.
        self._monitor_target = {"running": False}
        self._monitor_result = None
        self._monitor_inflight = False

        from PySide6.QtCore import QTimer
        self._status_timer = QTimer(self)
        interval = load_config(base_dir()).get("monitor_interval_ms", 2000)
        self._status_timer.setInterval(interval)
        self._status_timer.timeout.connect(self.update_status)
        self._status_timer.start()

        if load_config(base_dir()).get("update_check", True):
            self._update_timer = QTimer(self)
            self._update_timer.setSingleShot(True)
            self._update_timer.timeout.connect(self.run_update_check)
            self._update_timer.start(3000)

    def _profile_name(self) -> str:
        """Current profile name from the Name field, falling back to a default
        so the container name / preview are never empty."""
        return self.name_edit.text().strip() or "New Profile"

    def _container_name(self) -> str:
        return f"llama-{slugify(self._profile_name())}"

    def active_catalog(self) -> dict:
        """The settings that apply to the mode currently selected in the form.

        Router CLI args outrank every member's preset value, so a router must
        not be able to carry model-level settings; conversely --models-max on a
        single-model llama-server is rejected outright.
        """
        return (router_catalog() if self.mode_combo.currentData() == "router"
                else member_catalog())

    def _apply_mode_to_settings_form(self) -> None:
        active = self.active_catalog()
        for group, box in self._group_boxes.items():
            box.setVisible(False)
        for key, (form, widget) in self._setting_rows.items():
            visible = key in active
            index = form.getWidgetPosition(widget)[0]
            if index >= 0:
                form.setRowVisible(index, visible)
            if visible:
                self._group_boxes[CATALOG[key].group].setVisible(True)

    def _on_mode_changed(self, _index=0) -> None:
        is_router = self.mode_combo.currentData() == "router"
        # Hide the whole "Router members" form row (its QFormLayout label too),
        # not just the inner widgets -- otherwise an orphaned "Router members"
        # label lingers on the server-mode form.
        self._left_form.setRowVisible(self._members_row, is_router)
        self._apply_mode_to_settings_form()
        # A router has no model of its own; its members carry those fields.
        for w in (self.model_edit, self.mmproj_edit, self.draft_model_edit):
            w.setEnabled(not is_router)
        self.lora_panel.setEnabled(not is_router)
        self.detached_check.setVisible(not is_router)
        # The reusable API key + harness block + status/exposure banner are
        # ROUTER-only concepts (a single server uses its own --api-key field in
        # the Settings column). Show them only in router mode.
        self.api_key_box.setVisible(is_router)
        self.harness_box.setVisible(is_router)
        self.configure_status.setVisible(is_router)
        self.monitor_status.setVisible(is_router)
        self._sync_load_mode_legacy()
        self.refresh_preview()
        if is_router:
            self.refresh_router_panel_header()

    def _sync_load_mode_legacy(self) -> None:
        """Gray out --no-mmap/--mlock when load-mode is set (it wins in argv)."""
        lm = self._widgets.get("load-mode")
        if lm is None:
            return
        load_mode_active = lm.value() != CATALOG["load-mode"].default
        for key in ("no-mmap", "mlock"):
            w = self._widgets.get(key)
            if w is not None:
                w.setEnabled(not load_mode_active)

    def _on_tab_changed(self, _index: int) -> None:
        # Entering the Configure tab must show a live key even for an edited-
        # but-unsaved router; refresh_router_panel_header() no-ops for
        # non-routers.
        if self.tabs.currentWidget() is self._configure_tab:
            self.refresh_router_panel_header()
        # Command preview / api-key / harness only make sense while configuring,
        # so hide the bottom strip on the Monitor/Benchmark tabs.
        self._config_bottom.setVisible(self.tabs.currentWidget() is self._configure_tab)

    def _on_stats_visibility(self, visible: bool) -> None:
        # Keep the toolbar button in sync when the dock is closed via its own X,
        # and persist the state.
        if self.stats_toggle_btn.isChecked() != visible:
            self.stats_toggle_btn.setChecked(visible)
        if visible:
            self._start_stats_worker()
        else:
            self._stop_stats_worker()
        self._save_stats_config()

    def _refresh_stats_target(self) -> None:
        # Read GUI/profile state on the UI thread only; the worker reads the
        # resulting plain tuple (safe under the GIL), never the widgets.
        self._stats_target = (self._monitored_container_name(),
                              self.current_profile().runtime.binary)

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

        self._stats_worker = StatsWorker(_build, interval_ms=1000, parent=self)
        self._stats_worker.sampled.connect(self.stats_panel.update_stats)
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
        cfg = load_config(base_dir())
        cfg["stats_open"] = self.stats_dock.isVisibleTo(self)
        cfg["stats_width"] = self.stats_dock.width() or cfg.get("stats_width", 320)
        save_config(cfg, base_dir())

    def _add_member_item(self, member: RouterMember) -> None:
        from PySide6.QtWidgets import QTableWidgetItem
        row = self.members_list.rowCount()
        # itemChanged fires per setItem; without this the handler reads a
        # half-populated row and hits None cells.
        blocked = self.members_list.blockSignals(True)
        self.members_list.insertRow(row)

        name = QTableWidgetItem(member.profile)
        name.setFlags(name.flags() & ~Qt.ItemIsEditable)   # the profile is the identity
        self.members_list.setItem(row, 0, name)

        # Empty means "derive from the profile name"; show the derived value as a
        # placeholder so the harness-facing id is never a mystery.
        mid = QTableWidgetItem(member.model_id)
        mid.setToolTip(f"Empty = {member_model_id(member)}")
        self.members_list.setItem(row, 1, mid)

        load = QTableWidgetItem()
        load.setFlags((load.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable)
        load.setCheckState(Qt.Checked if member.load_on_startup else Qt.Unchecked)
        self.members_list.setItem(row, 2, load)

        self.members_list.setItem(row, 3, QTableWidgetItem(str(member.stop_timeout)))
        self.members_list.blockSignals(blocked)

    def set_member_fields(self, row: int, model_id: str | None = None,
                          load_on_startup: bool | None = None,
                          stop_timeout: int | None = None) -> None:
        """Programmatic equivalent of editing a member row (used by tests)."""
        if model_id is not None:
            self.members_list.item(row, 1).setText(model_id)
        if load_on_startup is not None:
            self.members_list.item(row, 2).setCheckState(
                Qt.Checked if load_on_startup else Qt.Unchecked)
        if stop_timeout is not None:
            self.members_list.item(row, 3).setText(str(stop_timeout))
        self.refresh_preview()

    def _member_candidates(self) -> list[str]:
        """Non-router profiles eligible to be added as router members.

        Read fresh from disk so a profile saved this session shows up without a
        restart. A router can't be a member, so router-mode profiles are
        filtered out -- that filter also excludes the router being edited once
        it's saved. We deliberately do NOT exclude by the Name field's current
        text: the natural add-a-member flow (switch to server mode, save the new
        model, switch back to router mode) leaves the new model's name in the
        Name field, and excluding it hid the very member the user just made until
        an app restart.
        """
        return [p.name for p in list_profiles(base_dir()) if p.mode != "router"]

    def _on_add_member(self) -> None:
        names = self._member_candidates()
        if not names:
            QMessageBox.information(self, "No profiles",
                                    "Save a model profile first; routers serve members.")
            return
        name, ok = QInputDialog.getItem(self, "Add member", "Profile:", names, 0, False)
        if ok and name:
            self._add_member_item(RouterMember(profile=name))
            self.refresh_preview()

    def _on_remove_member(self) -> None:
        for row in sorted({i.row() for i in self.members_list.selectedIndexes()},
                          reverse=True):
            self.members_list.removeRow(row)
        self.refresh_preview()

    def _has_unsaved_changes(self) -> bool:
        # Stateless dirty check: the form is clean iff current_profile() round-trips
        # equal to the stored profile of the same name (verified: dataclass equality,
        # dict-order-independent). A never-saved profile counts as changed.
        cur = self.current_profile()
        saved = {p.name: p for p in list_profiles(base_dir())}.get(cur.name)
        return saved is None or cur != saved

    def _on_edit_member(self) -> None:
        row = self.members_list.currentRow()
        if row < 0:
            return
        item = self.members_list.item(row, 0)   # column 0 = member profile name
        if item is None:
            return
        name = item.text()
        target = {p.name: p for p in list_profiles(base_dir())}.get(name)
        if target is None:
            QMessageBox.warning(
                self, "Profile missing",
                f"Profile '{name}' no longer exists — remove it from the router "
                f"or recreate it.")
            return
        if self._has_unsaved_changes():
            choice = QMessageBox.question(
                self, "Unsaved changes",
                f"You have unsaved changes to '{self.current_profile().name}'. "
                f"Editing member '{name}' will load its profile and lose those changes.",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Cancel)
            if choice == QMessageBox.Cancel:
                return
            if choice == QMessageBox.Save:
                self.save_current_profile()
        self.load_profile(target)

    def members(self) -> list:
        out = []
        for row in range(self.members_list.rowCount()):
            name = self.members_list.item(row, 0)
            mid = self.members_list.item(row, 1)
            load = self.members_list.item(row, 2)
            timeout_item = self.members_list.item(row, 3)
            if name is None:
                continue          # row still being built
            try:
                timeout = int(((timeout_item.text() if timeout_item else "") or "10").strip())
            except ValueError:
                timeout = 10
            out.append(RouterMember(
                profile=name.text(),
                model_id=((mid.text() if mid else "") or "").strip(),
                load_on_startup=bool(load is not None and load.checkState() == Qt.Checked),
                stop_timeout=timeout,
            ))
        return out

    def member_pairs(self) -> list:
        """(RouterMember, member Profile) pairs for members whose profile exists."""
        return resolve_member_pairs(self.members(), base_dir())

    def missing_member_profiles(self) -> list:
        """Member profile names that no longer exist on disk.

        Dropping these silently launched a router serving fewer models than the
        list showed, and any harness pinned to the missing id got a 404 with
        nothing in the GUI explaining why."""
        by_name = {p.name: p for p in list_profiles(base_dir())}
        return [m.profile for m in self.members() if m.profile not in by_name]

    def router_issues(self) -> list:
        """Validation issues for the current profile, router context included."""
        p = self.current_profile()
        key_present = bool(api_key_store.resolve_api_key(self.router_base_dir(), p)) \
            if p.mode == "router" else False
        issues = validate(p, binary_found=runtime.binary_available(p.runtime.binary),
                          members=self.member_pairs(), api_key_present=key_present)
        for name in self.missing_member_profiles():
            issues.append(Issue(
                "error",
                f"Member profile {name!r} no longer exists; it would be dropped from "
                f"the router silently. Remove it or recreate the profile."))
        return issues

    def _field_with_browse(self, line_edit: QLineEdit, dot: QWidget | None = None) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(line_edit, 1)
        if dot is not None:
            row.addWidget(dot)
        btn = QPushButton("Browse…")
        btn.clicked.connect(lambda: self._browse_into(line_edit))
        row.addWidget(btn)
        return container

    def _browse_into(self, line_edit: QLineEdit) -> None:
        start = os.path.expanduser("~")
        for m in self.mounts_panel.mounts():
            if m.host:
                start = m.host
                break
        path, _ = QFileDialog.getOpenFileName(self, "Select file", start)
        if not path:
            return
        container_path = host_to_container(path, self.mounts_panel.mounts())
        if container_path is None:
            QMessageBox.warning(
                self, "File not in a mounted folder",
                "The selected file is not inside any mounted folder.\n"
                "Add a folder mount that contains it first, then pick it again.")
            return
        line_edit.setText(container_path)

    def load_profile(self, p: Profile) -> None:
        self._stop_log_follower()
        self._profile = p
        self.name_edit.setText(p.name)
        self.image_edit.setText(p.image)
        self.model_edit.setText(p.model)
        self.binary_combo.setCurrentText(p.runtime.binary)
        self.gpu_combo.setCurrentIndex(max(0, self.gpu_combo.findData(p.runtime.gpu_mode)))
        self.mounts_panel.set_mounts(p.mounts)
        self.mmproj_edit.setText(p.mmproj or "")
        self.draft_model_edit.setText(p.draft_model or "")
        self.lora_panel.set_loras(p.loras)
        self.raw_edit.setText(p.raw_args)
        self.extra_args_edit.setText(p.runtime.extra_run_args)
        self.selinux_check.setChecked(p.runtime.selinux_label_disable)
        self.detached_check.setChecked(p.runtime.detached)
        for key, w in self._widgets.items():
            w.set_value(w.setting.default)
            if key in p.settings:
                w.set_value(p.settings[key])
        index = self.mode_combo.findData(p.mode)
        self.mode_combo.setCurrentIndex(index if index >= 0 else 0)
        self.bind_host_combo.setCurrentText(p.runtime.bind_host)
        self._router_statuses = {}
        self._spec_prev = None
        self._props = None
        self._props_model = None
        self.members_list.setRowCount(0)
        for member in p.members:
            self._add_member_item(member)
        self._on_mode_changed()
        self.apply_model_caps()
        self.refresh_preview()
        # Reattach-after-restart is the common path: selecting a saved router
        # must show its key, harness block and exposure banner immediately.
        self.api_key_box.set_scope(p.runtime.router_key_mode)
        self.refresh_router_panel_header()

    def current_profile(self) -> Profile:
        # Filtering lives here, in the UI, per the project's catalog contract --
        # never in command_builder. A value left over from the other mode (or
        # loaded from older profile JSON) must not reach argv.
        active = self.active_catalog()
        settings = {}
        for key, w in self._widgets.items():
            if key in active and w.is_set():
                settings[key] = w.value()
        # port is always stored
        settings["port"] = self._widgets["port"].value()
        return Profile(
            name=self._profile_name(),
            image=self.image_edit.text(),
            runtime=Runtime(binary=self.binary_combo.currentText(),
                            gpu_mode=self.gpu_combo.currentData(),
                            selinux_label_disable=self.selinux_check.isChecked(),
                            extra_run_args=self.extra_args_edit.text(),
                            bind_host=self.bind_host_combo.currentText().strip()
                                      or "127.0.0.1",
                            detached=self.detached_check.isChecked(),
                            router_key_mode=self.api_key_box._current_scope()),
            mounts=self.mounts_panel.mounts(),
            model=self.model_edit.text(),
            mmproj=self.mmproj_edit.text() or None,
            draft_model=self.draft_model_edit.text() or None,
            loras=self.lora_panel.loras(),
            settings=settings,
            raw_args=self.raw_edit.text(),
            mode=self.mode_combo.currentData() or "server",
            members=self.members(),
        )

    def build_current_command(self, p: Profile | None = None) -> list:
        """argv for the current profile, including the router mount.

        Every path that renders a command (preview, Export .sh, the diagnostic
        report, launch) must go through here: build_command() omits the
        -v <dir>:/router:ro mount unless it is told the host directory, and a
        router command without that mount cannot start -- its --models-preset
        and --api-key-file paths would not exist inside the container.
        """
        p = p or self.current_profile()
        if p.mode != "router":
            return build_command(p, detach=p.runtime.detached)
        return build_command(
            p, router_host_dir=str(api_key_store.router_dir(self.router_base_dir(), p.name)))

    def preview_text(self) -> str:
        return " ".join(self.build_current_command())

    def refresh_preview(self) -> None:
        self.preview.setPlainText(self.preview_text())

    def _reload_profile_list(self):
        self.profile_combo.clear()
        self._profiles = {p.name: p for p in list_profiles(base_dir())}
        self.profile_combo.addItems(list(self._profiles.keys()))

    def _on_pick_profile(self, _index):
        self._stop_log_follower()
        name = self.profile_combo.currentText()
        if name in self._profiles:
            self.load_profile(self._profiles[name])

    def save_current_profile(self):
        p = self.current_profile()       # name comes from the Name field
        self._profile = p
        save_profile(p, base_dir())
        self._reload_profile_list()
        self.profile_combo.setCurrentText(p.name)

    def save_as_profile(self):
        name, ok = QInputDialog.getText(self, "Save As", "Profile name:",
                                        text=self._profile_name())
        if ok and name:
            self.name_edit.setText(name)
            self.save_current_profile()

    def delete_current_profile(self):
        name = self.profile_combo.currentText()
        if name:
            delete_profile(name, base_dir())
            self._reload_profile_list()

    def router_base_dir(self):
        return base_dir()

    def router_api_key(self) -> str:
        p = self.current_profile()
        return (api_key_store.resolve_api_key(self.router_base_dir(), p)
                or api_key_store.ensure_api_key(self.router_base_dir(), p.name))

    def prepare_router_files(self) -> tuple:
        """Write models.ini + api-key for the current router. Returns (dir, warnings)."""
        name = self._profile_name()
        result = render_preset(self.member_pairs())
        api_key_store.prepare_launch_key(self.router_base_dir(), self.current_profile())
        api_key_store.write_preset(self.router_base_dir(), name, result.text)
        return str(api_key_store.router_dir(self.router_base_dir(), name)), result.warnings

    def _on_key_scope_changed(self, mode: str) -> None:
        # The radio already feeds current_profile(); persist and re-resolve display.
        self.save_current_profile()
        self.refresh_router_panel_header()
        self._notify_key_change_needs_relaunch()

    def _on_key_saved(self, scope: str, value: str) -> None:
        base = self.router_base_dir()
        if scope == "global":
            api_key_store.write_global_key(base, value)
        else:
            api_key_store.set_profile_key(base, self._profile_name(), value)
        self.refresh_router_panel_header()
        self._notify_key_change_needs_relaunch()

    def _notify_key_change_needs_relaunch(self) -> None:
        """The running router keeps serving the key it launched with -- a key
        change here only takes effect on the NEXT launch. Without this, a user
        who copies the newly-shown key into their harness while the router is
        still up gets a bare 401 with nothing in the GUI explaining why."""
        p = self.current_profile()
        if runtime.container_state(self._container_name(),
                                   p.runtime.binary) == "running":
            QMessageBox.information(
                self, "Relaunch needed",
                "The router is running. Relaunch it for the new API key to "
                "take effect.")

    def _set_router_connected(self, connected: bool) -> None:
        self.configure_status.set_connected(connected)
        self.monitor_status.set_connected(connected)

    def _set_router_error(self, text: str) -> None:
        self.configure_status.set_error(text)
        self.monitor_status.set_error(text)

    def _set_router_exposure(self, text: str) -> None:
        self.configure_status.set_exposure_warning(text)
        self.monitor_status.set_exposure_warning(text)

    def refresh_router_panel_header(self) -> None:
        p = self.current_profile()
        if p.mode != "router":
            # Clear relocated router state so a previous router's exposure
            # banner, API key, and harness endpoint don't linger on the
            # Configure/Monitor tabs after switching to an unrelated profile.
            self._set_router_exposure("")
            self.api_key_box.set_key("")
            self.harness_box.harness_text.setPlainText("")
            return
        host = p.runtime.bind_host
        display_host = dial_host(host)
        port = p.settings.get("port", 8080)
        # A router without a key is unusable, and the harness block exists so the
        # key can be copied BEFORE the first launch. Generating here is
        # idempotent and is the only side effect on this path.
        key = (api_key_store.resolve_api_key(self.router_base_dir(), p)
               or api_key_store.ensure_api_key(self.router_base_dir(), p.name))
        self.api_key_box.set_key(key)
        self.harness_box.set_endpoint(
            f"http://{display_host}:{port}",
            [member_model_id(m) for m in p.members])
        self._set_router_exposure(
            f"Bound to {host}: reachable beyond this machine. The API key is required."
            if host not in LOOPBACK_HOSTS else "")

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
            self.monitor_panel.set_draft_stats(spec_delta(self._spec_prev, cur),
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
            return api_key_store.read_api_key(self.router_base_dir(), p.name)
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
        self.monitor_panel.set_props(info)
        return model_key

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
        self.status_label.setText("● failed to start")
        reason = (f"launch failed: {text.splitlines()[-1][:200]}"
                  if text else "launch failed")
        self._set_router_error(reason)
        if show_dialog:
            QMessageBox.critical(self, "Launch failed", reason)

    def adopt_running_containers(self) -> list:
        """Containers this launcher owns, so a detached router survives a GUI restart."""
        p = self.current_profile()
        return runtime.list_launcher_containers(p.runtime.binary)

    def _router_host(self, p: Profile) -> str:
        """The address the GUI itself dials for this profile."""
        return dial_host(p.runtime.bind_host)

    def refresh_router_models(self) -> None:
        p = self.current_profile()
        if p.mode != "router":
            return
        host = self._router_host(p)
        port = p.settings.get("port", 8080)
        key = api_key_store.read_api_key(self.router_base_dir(), p.name)
        models = router_api.list_models(host, port, key)
        if models is None:            # unreachable, as opposed to serving nothing
            self._router_statuses = {}
            self.router_models_table.set_models([])
            self._set_router_connected(False)
            return
        self._router_statuses = {m.id: m.status for m in models}
        self.router_models_table.set_models(models)
        self._set_router_connected(True)

    def _router_pollable_model(self) -> str | None:
        """The resident model to scope Monitor polling to, or None.

        Sleeping and unloaded models are skipped: there is nothing to measure,
        and not polling them is the whole point of an idle-unloading host."""
        for model_id, status in self._router_statuses.items():
            if status == "loaded":
                return model_id
        return None

    def _on_router_load(self, model_id: str) -> None:
        p = self.current_profile()
        key = api_key_store.read_api_key(self.router_base_dir(), p.name)
        ok = router_api.load_model(self._router_host(p), p.settings.get("port", 8080),
                                   key, model_id)
        self.refresh_router_models()
        if not ok:
            # Silently discarding this left a failed load looking identical to a
            # slow one: the row just stayed "unloaded" forever.
            self._set_router_error(f"load failed: {model_id}")

    def _on_router_unload(self, model_id: str) -> None:
        p = self.current_profile()
        key = api_key_store.read_api_key(self.router_base_dir(), p.name)
        ok = router_api.unload_model(self._router_host(p), p.settings.get("port", 8080),
                                     key, model_id)
        self.refresh_router_models()
        if not ok:
            self._set_router_error(f"unload failed: {model_id}")

    def _validate_or_warn(self) -> bool:
        p = self.current_profile()
        # The key must exist before the exposure rule is evaluated: a router
        # always gets one at launch, but this runs before prepare_router_files.
        if p.mode == "router":
            api_key_store.prepare_launch_key(self.router_base_dir(), p)
        issues = self.router_issues()
        errors = [i for i in issues if i.level == "error"]
        if errors:
            QMessageBox.critical(self, "Cannot launch",
                                 "\n".join(i.message for i in errors))
            return False
        warns = [i for i in issues if i.level == "warning"]
        if warns:
            QMessageBox.warning(self, "Warnings", "\n".join(i.message for i in warns))
        return True

    def vram_check(self) -> str | None:
        p = self.current_profile()
        meta, weights, _caps = model_info.inspect_model(p.model, self.mounts_panel.mounts()) if p.model else (None, None, None)
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

    def on_launch(self):
        if not self._validate_or_warn():
            return
        p = self.current_profile()

        if p.mode == "router":
            router_host_dir, warnings = self.prepare_router_files()
            if warnings:
                QMessageBox.warning(self, "Preset warnings", "\n".join(warnings))
            argv = build_command(p, router_host_dir=router_host_dir)
            # Relaunching over a LIVE router would drop a resident model and any
            # in-flight harness requests, so confirm before tearing it down.
            if runtime.container_state(self._container_name(),
                                       p.runtime.binary) == "running":
                answer = QMessageBox.question(
                    self, "Router already running",
                    "This router is already running. Relaunching stops it, "
                    "unloading any resident model and dropping in-flight "
                    "requests. Continue?")
                if answer != QMessageBox.Yes:
                    return

            # A stopped container of the same name would block the new run,
            # since router mode deliberately omits --rm. Chain rather than fire
            # both at once: _spawn_async is asynchronous, so an unchained run
            # would race the removal and lose with "name already in use".
            self.monitor_panel.reset()
            self.benchmark_panel.reset()
            self.benchmark_panel.set_benchmark_history(
                benchmark_store.load(default_base_dir(), p.name))
            self._spec_prev = None
            self._props = None
            self._props_model = None
            self._spawn_async(
                runtime.rm_argv(self._container_name(), p.runtime.binary),
                on_done=lambda: self._spawn_async(
                    argv, on_done=self.update_status,
                    # Detached means no terminal, so a bad image ref or a CDI
                    # failure would otherwise produce nothing but a status label
                    # stuck on "stopped".
                    on_error=self._report_launch_error))
            self.refresh_router_panel_header()
            return

        warn = self.vram_check()
        if warn:
            QMessageBox.warning(self, "VRAM check", warn)
        self.monitor_panel.reset()
        self.benchmark_panel.reset()
        self.benchmark_panel.set_benchmark_history(
            benchmark_store.load(default_base_dir(), p.name))
        self.monitor_panel.set_endpoints(
            p.settings.get("port", 8080),
            bool(p.settings.get("embeddings")),
            bool(p.settings.get("reranking")),
        )
        if p.runtime.detached:
            argv = build_command(p, detach=True)
            # Detached drops --rm, so a stale stopped container of this name
            # would block the run with "name already in use". Remove it first,
            # then chain the run (mirrors the router branch above). on_error
            # surfaces bad image / CDI / flag failures the terminal used to
            # show -- show_dialog is fixed here, at launch time, so a later
            # profile/mode switch before the error fires can't change it.
            self._spawn_async(
                runtime.rm_argv(self._container_name(), p.runtime.binary),
                on_done=lambda: self._spawn_async(
                    argv, on_done=self.update_status,
                    on_error=lambda e=None: self._report_launch_error(
                        e, show_dialog=True)))
        else:
            argv = build_command(p)
            terminal.launch(argv)
        # Don't attach the log follower here: the container is created
        # asynchronously and doesn't exist yet. update_status() starts the
        # follower once it is actually running (podman logs replays from the
        # start, so no early output is missed).

    def _spawn_async(self, argv: list[str], on_done=None, on_error=None):
        """Run argv in the background via QProcess so the UI thread never blocks.
        Calls on_done() when the process finishes.

        on_error(text) receives podman's stderr on a non-zero exit, and a
        message if the binary could not be started at all — `finished` is NOT
        emitted on FailedToStart, so without errorOccurred a launch that never
        began would report nothing whatsoever.
        """
        from PySide6.QtCore import QProcess
        proc = QProcess(self)
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
        # Stop the log follower immediately; run `podman stop` asynchronously so a
        # slow stop (podman waits up to its grace period for SIGTERM) never freezes
        # the GUI — which previously made Stop look like it did nothing.
        self._stop_log_follower()
        p = self.current_profile()
        self.status_label.setText("● stopping…")
        argv = runtime.stop_argv(self._container_name(), p.runtime.binary)
        self._stop_proc = self._spawn_async(argv, on_done=self.update_status)

    def on_restart(self):
        self._stop_log_follower()
        p = self.current_profile()
        self.status_label.setText("● restarting…")
        argv = runtime.stop_argv(self._container_name(), p.runtime.binary)
        # Launch only after the stop completes, so the new container's --name/port
        # don't collide with the one being torn down.
        self._stop_proc = self._spawn_async(argv, on_done=self.on_launch)

    def _on_enable_metrics(self):
        self._widgets["metrics"].set_value(True)
        self.on_restart()

    def _resolve_benchmark_member(self, p: Profile, model_scope: str | None):
        """The child Profile for a router's loaded model, for build_snapshot().

        build_snapshot() requires a Profile (it reads .settings/.model), never
        a RouterMember -- a RouterMember has neither and raises AttributeError.
        Falling back to None (profile.settings, i.e. the router's own -- empty
        -- settings) is a lesser-quality snapshot but never crashes.
        """
        if p.mode != "router" or model_scope is None:
            return None
        for member, member_profile in self.member_pairs():
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
                                        self._poll_api_key(p), None, True)
        if p.mode == "router":
            host = self._router_host(p)
            model_scope = self._router_pollable_model()
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
        p = self.current_profile()
        prepared = self._prepare_benchmark(p)
        if prepared is None:
            self.benchmark_panel.set_benchmark_progress("No model loaded to benchmark.")
            return
        client, snapshot = prepared
        self._benchmark_profile_name = p.name
        self.benchmark_panel.set_benchmark_running(True)
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
        p = self.current_profile()
        prepared = self._prepare_benchmark(p)
        if prepared is None:
            self.benchmark_panel.set_benchmark_progress("No model loaded to benchmark.")
            return
        client, snapshot = prepared
        self._benchmark_profile_name = p.name
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")

        thread = QThread(self)
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
        self.benchmark_panel.set_benchmark_running(True)
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
        benchmark_store.clear(default_base_dir(), self.current_profile().name)
        self.benchmark_panel.reset()

    def _on_benchmark_finished(self, run) -> None:
        name = self._benchmark_profile_name or self.current_profile().name
        base = default_base_dir()
        previous_runs = benchmark_store.load(base, name)
        previous = previous_runs[-1] if previous_runs else None
        benchmark_store.append(base, name, run)
        run_dict = dataclasses.asdict(run)
        delta = benchmark_store.delta(run_dict, previous) if previous is not None else None
        self.benchmark_panel.show_benchmark_run(run_dict, delta)
        self.benchmark_panel.set_benchmark_history(benchmark_store.load(base, name))
        self.benchmark_panel.set_benchmark_running(False)

    def _on_benchmark_failed(self, msg: str) -> None:
        self.benchmark_panel.set_benchmark_progress(f"Benchmark failed: {msg}")
        self.benchmark_panel.set_benchmark_running(False)

    def on_fetch_latest(self):
        repo, tag = split_image(self.image_edit.text())
        if not repo:
            QMessageBox.information(
                self, "No image",
                "Set or Detect an image first — Fetch latest looks up the newest "
                "build for the image's repository.")
            return
        prefix = variant_prefix(tag) if tag else "server-cuda12"
        self._fetch_repo = repo
        self._fetch_got_result = False
        self.fetch_btn.setEnabled(False)
        self.fetch_btn.setText("Fetching…")
        self.update_badge.setEnabled(False)
        worker = _UpdateWorker(repo, prefix, parent=self)
        worker.found.connect(self._on_fetch_found)
        worker.failed.connect(self._on_fetch_failed)
        worker.finished.connect(self._on_fetch_finished)   # QThread built-in
        self._fetch_worker = worker
        worker.start()

    def _on_fetch_found(self, tag: str) -> None:
        self._fetch_got_result = True
        image = f"{self._fetch_repo}:{tag}"
        self.image_edit.setText(image)
        QMessageBox.information(
            self, "Latest build",
            f"Image set to {image}.\n\nThis only updates the tag — the build is NOT "
            f"downloaded. Pull it with:\n  podman pull {image}\n(or docker pull).")

    def _on_fetch_failed(self, msg: str) -> None:
        self._fetch_got_result = True
        QMessageBox.warning(
            self, "Fetch failed", f"Couldn't fetch the latest build:\n{msg}")

    def _on_fetch_finished(self) -> None:
        self.fetch_btn.setEnabled(True)
        self.fetch_btn.setText("Fetch latest")
        self.update_badge.setEnabled(True)
        if not self._fetch_got_result:
            QMessageBox.information(
                self, "Latest build", "No newer build found for this image.")

    def detect_image(self):
        binary = self.binary_combo.currentText()
        images = runtime.list_local_images(binary)
        if not images:
            QMessageBox.information(
                self, "Detect image",
                f"No local llama.cpp images found for '{binary}'.\n"
                f"Pull one (e.g. {binary} pull ghcr.io/ggml-org/llama.cpp:server) "
                f"or type the image reference yourself.")
            return
        if len(images) == 1:
            self.image_edit.setText(images[0])
            return
        choice, ok = QInputDialog.getItem(
            self, "Detect image", "Local llama.cpp images:", images, 0, False)
        if ok and choice:
            self.image_edit.setText(choice)

    def _autofill_image_if_empty(self):
        if self.image_edit.text().strip():
            return
        images = runtime.list_local_images(self.binary_combo.currentText())
        if len(images) == 1:
            self.image_edit.setText(images[0])

    def check_for_update(self, tags: list[str]) -> str | None:
        repo, tag = split_image(self.image_edit.text())
        if not tag:
            return None
        prefix = variant_prefix(tag)
        latest = registry.latest_build_tag(tags, prefix)
        if latest and latest != tag:
            return latest
        return None

    def run_update_check(self):
        repo, tag = split_image(self.image_edit.text())
        if not repo or not tag:
            return
        prefix = variant_prefix(tag)
        current_tag = tag

        def _on_found(latest: str):
            if latest != current_tag:
                m = registry._BUILD_RE.match(latest)
                build_id = f"b{m.group('num')}" if m else latest
                self.update_badge.setText(f"newer build {build_id} available")
                self.update_badge.setVisible(True)

        worker = _UpdateWorker(repo, prefix, parent=self)
        worker.found.connect(_on_found)
        self._update_worker = worker
        worker.start()

    def update_status(self):
        # Keep the (container_name, binary) snapshot StatsWorker reads current
        # while the dock is open and the user switches profile/instance --
        # this UI-thread timer callback is the only writer of _stats_target.
        if self._stats_worker is not None and self._stats_worker.isRunning():
            self._refresh_stats_target()
        p = self._monitored_profile()
        if not runtime.binary_available(p.runtime.binary):
            self.status_label.setText("● stopped")
            self.web_ui_btn.setEnabled(False)
            self.benchmark_panel.set_benchmark_available(False)
            self._monitor_target = {"running": False}
            self._monitor_result = None
            return
        name = self._monitored_container_name()
        state = runtime.container_state(name, p.runtime.binary)
        # Default the gather target to "don't poll"; the running branch below
        # overwrites it with the live snapshot. Nothing is gathered off-thread
        # until update_status confirms the container is up. (_monitor_result is
        # left intact here so the running branch can render the last gather;
        # it's cleared below only when nothing is running.)
        self._monitor_target = {"running": False}
        hstatus = health.probe_health(p.settings.get("port", 8080),
                                     host=dial_host(p.runtime.bind_host)) \
            if state == "running" else "down"
        self.status_label.setText("● " + health.derive_status(state, hstatus))
        self.web_ui_btn.setEnabled(state == "running")
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
            self._monitor_target = self._compute_monitor_target(
                running=True, model_scope=router_model_key)
            if self._monitor_result is not None:
                self.monitor_panel.update_stats(self._monitor_result)
            if not self._monitor_inflight:
                self._monitor_inflight = True
                QThreadPool.globalInstance().start(
                    _MonitorGather(self, self._monitor_target))
            self._update_spec_stats(p)
        else:
            # Nothing running: drop the last gather so a stale summary isn't
            # rendered on the next start before a fresh gather completes.
            self._monitor_result = None
        if p.mode == "router":
            if state == "running":
                self.refresh_router_models()
            else:
                # Router stopped/removed: clear the stale model list + connected
                # state so a dead router doesn't keep showing load/unload rows.
                self._router_statuses = {}
                self.router_models_table.set_models([])
                self._set_router_connected(False)
        # router_model_key was resolved from _refresh_props above (when ready)
        # rather than polled again here, so this reuses that single call to
        # _router_pollable_model() instead of doubling it.
        ready = state == "running" and hstatus == "ready"
        if ready and p.mode == "router":
            ready = router_model_key is not None
        self.benchmark_panel.set_benchmark_available(ready)
        self._refresh_instances_list()

    def _refresh_instances_list(self) -> None:
        binary = self.current_profile().runtime.binary
        self._instances = build_instances(
            runtime.list_launcher_containers(binary), list_profiles(base_dir()))
        rows = []
        for inst in self._instances:
            summ = self.instance_summary(inst)
            rows.append({"name": inst.name, "profile": inst.profile, "port": inst.port,
                         "running": inst.running, "health": summ["health"],
                         "stat": summ["stat"]})
        self.monitor_panel.set_instances(rows, self._monitored_container_name())

    def _on_instance_selected(self, name: str) -> None:
        inst = next((i for i in self._instances if i.name == name), None)
        # Selecting the form's own container means "monitor the current profile" (fallback).
        self._active_instance = None if (inst is None or name == self._container_name()) else inst
        self._start_log_follower()          # retarget the follower at the new container
        self.update_status()

    def _on_instance_stop(self, name: str) -> None:
        binary = self.current_profile().runtime.binary
        self._spawn_async(runtime.stop_argv(name, binary), on_done=self.update_status)
        if self._active_instance is not None and self._active_instance.name == name:
            self._active_instance = None

    def _on_instance_remove(self, name: str) -> None:
        # A stopped launcher container lingers in `podman ps -a` with no useful
        # action; remove it so the instances list can be cleared.
        binary = self.current_profile().runtime.binary
        self._spawn_async(runtime.rm_argv(name, binary), on_done=self.update_status)
        if self._active_instance is not None and self._active_instance.name == name:
            self._active_instance = None

    def _monitored_profile(self) -> Profile:
        inst = self._active_instance
        if inst is None:
            return self.current_profile()
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
            runtime=Runtime(bind_host=inst.host),
            settings={"port": inst.port or 8080,
                      "embeddings": inst.embeddings, "reranking": inst.reranking,
                      "metrics": bool(stored.settings.get("metrics")) if stored else False},
        )

    def _monitored_container_name(self) -> str:
        return self._active_instance.name if self._active_instance else self._container_name()

    def _instance_api_key(self, inst) -> str | None:
        """API key for polling a specific running instance -- the key that
        instance's server uses (mirrors _poll_api_key, but keyed off the
        Instance since the summary runs per-row). A router reads its key store;
        a server uses its stored --api-key. Without this, /metrics polls went
        out unauthenticated and the auth middleware answered 401 every tick."""
        if inst.mode == "router":
            return api_key_store.read_api_key(self.router_base_dir(), inst.profile)
        stored = next((p for p in list_profiles(base_dir())
                       if p.name == inst.profile), None)
        return (stored.settings.get("api-key") or None) if stored else None

    def instance_summary(self, inst) -> dict:
        if not inst.running or inst.port is None:
            return {"running": inst.running, "health": "down", "stat": ""}
        hstatus = health.probe_health(inst.port, host=inst.host)
        if inst.embeddings or inst.reranking:
            stat = "ready" if hstatus == "ready" else ""
        else:
            tok = metrics.fetch_metrics(inst.port, host=inst.host,
                                        api_key=self._instance_api_key(inst)).get(
                "llamacpp:predicted_tokens_seconds")
            stat = f"{tok:.0f} tok/s" if tok else ("ready" if hstatus == "ready" else "")
        return {"running": True, "health": hstatus, "stat": stat}

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
        host, key, ms, poll = (dial_host(p.runtime.bind_host),
                               self._poll_api_key(p), None, True)
        if p.mode == "router":
            host = self._router_host(p)
            ms = model_scope
            poll = ms is not None
        return {
            "running": True,
            "port": p.settings.get("port", 8080),
            "metrics_on": bool(p.settings.get("metrics")),
            "host": host, "key": key, "model_scope": ms, "poll": poll,
            "name": self._monitored_container_name(),
            "binary": p.runtime.binary,
        }

    def collect_monitor_data(self) -> dict:
        """Gather the Monitor summary synchronously (used by tests and any
        direct caller). The live poll instead snapshots _compute_monitor_target()
        and lets the worker call build_monitor_data() off the UI thread."""
        p = self._monitored_profile()
        ms = self._router_pollable_model() if p.mode == "router" else None
        return build_monitor_data(
            self._compute_monitor_target(running=True, model_scope=ms)) or {}

    def _log_follower_active(self) -> bool:
        from PySide6.QtCore import QProcess
        return (self._log_proc is not None
                and self._log_proc.state() != QProcess.NotRunning)

    def _start_log_follower(self):
        from PySide6.QtCore import QProcess
        self._stop_log_follower()
        p = self._monitored_profile()
        name = self._monitored_container_name()
        # Attaching `podman logs -f` before the container exists just prints
        # "no such container" and exits, stranding the logs pane on that error.
        # Skip until it exists; update_status() retries once it's running and
        # `podman logs` replays from the beginning, so no early output is lost.
        if not runtime.container_exists(name, p.runtime.binary):
            return
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(
            lambda: self._enqueue_log(bytes(proc.readAllStandardOutput()).decode("utf-8", "replace")))
        argv = runtime.logs_argv(name, p.runtime.binary)
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
        self.monitor_panel.append_log(text)

    def _stop_log_follower(self):
        # Stop the flush timer and drop any buffered tail so switching/stopping a
        # container can't leak one container's trailing lines into the next.
        self._log_flush_timer.stop()
        self._log_pending.clear()
        if self._log_proc is not None:
            self._log_proc.kill()
            self._log_proc = None

    def _stop_timers(self) -> None:
        """Stop background timers so a torn-down window stops firing update_status
        (and any pending update check). Idempotent — QTimer.stop() on a stopped
        timer is a no-op. _update_timer exists only when update_check is enabled.
        Also drains the stats worker via _stop_stats_worker() so a torn-down
        window can't leave a StatsWorker QThread running.

        Also tears down a running benchmark thread: cancels the worker (so a
        loop mid-repeat unwinds instead of running to completion against a
        closed window) and waits for the QThread to actually stop, since a
        Python interpreter shutdown with a live QThread can abort/crash.

        worker.finished/failed are already wired to thread.quit() (see
        _on_benchmark_run), but that's a queued cross-thread connection: it
        only takes effect once THIS (UI) thread's event loop runs and
        delivers it. A bare wait() never pumps events, so it would deadlock
        waiting for a quit() that never arrives; pump events between short
        waits instead. terminate() is a last-resort backstop so a stuck
        worker can never block window/app teardown indefinitely.
        """
        self._status_timer.stop()
        update_timer = getattr(self, "_update_timer", None)
        if update_timer is not None:
            update_timer.stop()
        self._stop_stats_worker()
        # Let any in-flight monitor gather finish (bounded) so it isn't writing
        # to the window during teardown; the pool's threads outlive the window,
        # so there's nothing to abort even if this times out.
        QThreadPool.globalInstance().waitForDone(3000)

        # Drain any in-flight registry-fetch / update-check QThread so closing
        # the window mid-fetch can't destroy a running QThread (abort/crash).
        # _UpdateWorker.run() is a blocking network call with no cancel flag, so
        # wait with a ceiling and terminate() as a last-resort backstop.
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

    def _on_export_sh(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export shell script", "run.sh",
                                              "Shell scripts (*.sh);;All files (*)")
        if path:
            self.export_sh(path)

    def open_web_ui(self):
        p = self.current_profile()
        port = p.settings.get("port", 8080)
        try:
            subprocess.Popen(["xdg-open", f"http://{dial_host(p.runtime.bind_host)}:{port}"],
                             start_new_session=True)
        except OSError:
            QMessageBox.warning(self, "Open Web UI", "Could not open browser (xdg-open not found).")

    def export_sh(self, path: str):
        cmd = " ".join(self.build_current_command())
        Path(path).write_text(f"#!/usr/bin/env bash\n{cmd}\n")
        os.chmod(path, 0o755)

    def gather_report_data(self) -> dict:
        import platform, json as _json
        p = self.current_profile()
        cmd = " ".join(self.build_current_command(p))
        # Pass the router context, or the report claims a healthy router has no
        # members and is exposed without a key -- in the one artifact users
        # paste when asking for help.
        issues = validate(p, binary_found=runtime.binary_available(p.runtime.binary),
                          members=self.member_pairs(),
                          api_key_present=bool(
                              api_key_store.resolve_api_key(self.router_base_dir(), p))
                          if p.mode == "router" else False)
        gpus = gpu.query_gpus()
        gpu_txt = "\n".join(f"{g.name}: {g.mem_used_mib}/{g.mem_total_mib} MiB, "
                            f"util {g.util_pct}%, {g.temp_c}C" for g in gpus) or "(no nvidia-smi)"
        runtime_txt = (f"binary={p.runtime.binary} gpu_mode={p.runtime.gpu_mode}\n"
                       f"rootless={runtime.is_rootless(p.runtime.binary)}\n"
                       f"{gpu_txt}\nOS={platform.platform()}")
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return {
            "generated_at": ts,
            "command": report_mod.redact_secrets(cmd),
            "profile": report_mod.redact_secrets(_json.dumps(profile_to_dict(p), indent=2)),
            "validation": [f"[{i.level}] {i.message}" for i in issues],
            "status_history": [self.status_label.text()],
            "runtime": runtime_txt,
            "metrics": self._metrics_report_text(p),
            "image": p.image,
            "logs": report_mod.redact_secrets(self.monitor_panel.log_view.toPlainText()[-4000:]),
        }

    def _metrics_report_text(self, p: Profile) -> str:
        """Snapshot of the live /metrics endpoint for the diagnostic report.

        Returns a note (and makes no network call) when --metrics is off, so the
        report explains why throughput is missing instead of silently omitting it.
        """
        from llama_launcher.services.metrics import kv_ratio
        port = p.settings.get("port", 8080)
        if not p.settings.get("metrics"):
            return ("(--metrics not enabled in this profile — turn it on and relaunch "
                    "to capture tok/s and KV-cache usage here)")
        # Mirror collect_monitor_data's host/key/scope derivation: /metrics needs
        # the API key, and on a router it is per-model (?model=id) reached via the
        # router host. Without these the report's fetch 401'd (or returned nothing)
        # and always printed the "no metrics returned" note for routers.
        host = dial_host(p.runtime.bind_host)
        key = self._poll_api_key(p)
        model_scope = None
        if p.mode == "router":
            host = self._router_host(p)
            model_scope = self._router_pollable_model()
        m = metrics.fetch_metrics(port, model=model_scope, api_key=key, host=host)
        slots = metrics.fetch_slots(port, model=model_scope, api_key=key, host=host)
        if not m and not slots:
            scope = " (no model currently loaded on the router)" if (
                p.mode == "router" and model_scope is None) else ""
            return (f"(no metrics returned from http://{host}:{port}/metrics{scope} — "
                    "generate the report while the server is running with --metrics)")
        lines = []
        gen = m.get("llamacpp:predicted_tokens_seconds")
        if gen is not None:
            lines.append(f"generation: {gen:.2f} tok/s")
        prompt = m.get("llamacpp:prompt_tokens_seconds")
        if prompt is not None:
            lines.append(f"prompt: {prompt:.2f} tok/s")
        kv = kv_ratio(m, slots)
        if kv is not None:
            lines.append(f"KV cache usage: {kv * 100:.0f}%")
        if m:
            if lines:
                lines.append("")
            lines += [f"{k} {v:g}" for k, v in sorted(m.items())]
        return "\n".join(lines)

    def _save_report(self, md: str, ts: str | None = None) -> Path:
        if ts is None:
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        reports_dir = base_dir() / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        out = reports_dir / f"llama-launcher-report-{ts}.md"
        out.write_text(md)
        return out

    def on_generate_report(self):
        cfg = load_config(base_dir())
        initial = cfg.get("report_sections", {s: True for s in report_mod.REPORT_SECTIONS})
        dlg = ReportDialog(initial, self)
        if not dlg.exec():
            return
        sections = dlg.selected_sections()
        cfg["report_sections"] = sections
        save_config(cfg, base_dir())
        data = self.gather_report_data()
        md = report_mod.build_report(data, sections)
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(md)
        saved = self._save_report(md, data.get("generated_at"))
        QMessageBox.information(self, "Report saved", f"Report saved to:\n{saved}")

    def closeEvent(self, event):
        if getattr(self, "_really_quit", False) or not self._minimize_to_tray:
            self._stop_log_follower()
            self._stop_timers()
            event.accept()
            from PySide6.QtWidgets import QApplication
            QApplication.instance().quit()
        else:
            event.ignore()
            self.hide()

    def quit_app(self):
        self._really_quit = True
        self._stop_log_follower()
        self._stop_timers()
        from PySide6.QtWidgets import QApplication
        QApplication.instance().quit()

    def apply_model_caps(self) -> None:
        p = self.current_profile()
        meta, size, caps = (None, None, None)
        if p.model:
            meta, size, caps = model_info.inspect_model(p.model, self.mounts_panel.mounts())
        self._last_caps = caps
        self.model_meta_label.setText(self._meta_caps_text(meta, size, caps))

        described = describe_relevance(caps) if caps else {}
        sugg_by_key, reason_by_key = self._suggestion_index(caps)   # concrete-value suggestions

        for key, widget in self._widgets.items():
            self._apply_dot(widget, key, described, sugg_by_key, reason_by_key)
        self._apply_field_dot(self._mmproj_dot, "mmproj", described, sugg_by_key, reason_by_key)
        self._apply_field_dot(self._draft_model_dot, "draft_model", described, sugg_by_key, reason_by_key)

    def _suggestion_index(self, caps):
        """key -> (Suggestion, reason). A multi-key suggestion indexes each key."""
        sugg_by_key, reason_by_key = {}, {}
        if not caps:
            return sugg_by_key, reason_by_key
        for sg in compute_suggestions(
                caps, self.current_profile().settings,
                mmproj_set=bool(self.mmproj_edit.text()),
                draft_set=bool(self.draft_model_edit.text())):
            for k in list(sg.settings) + list(sg.fields):
                sugg_by_key[k] = sg
                reason_by_key[k] = sg.text
        return sugg_by_key, reason_by_key

    @staticmethod
    def _dot_state_for(key, described, sugg_by_key, reason_by_key):
        tier, reason = described.get(key, (Tier.NEUTRAL, ""))
        state = {"recommended": "suggested", "tune": "suggested",
                 "na": "muted"}.get(getattr(tier, "value", ""), "none")
        return state, reason

    def _apply_dot(self, widget, key, described, sugg_by_key, reason_by_key) -> None:
        state, reason = self._dot_state_for(key, described, sugg_by_key, reason_by_key)
        on_apply = None
        if key in sugg_by_key:
            state = "suggested"
            reason = reason_by_key[key]
            sg = sugg_by_key[key]
            on_apply = lambda s=sg: self._apply_suggestion(s)
        widget.set_suggestion(state, reason, on_apply)

    def _apply_field_dot(self, dot, key, described, sugg_by_key, reason_by_key) -> None:
        state, reason = self._dot_state_for(key, described, sugg_by_key, reason_by_key)
        on_apply = None
        if key in sugg_by_key:
            state = "suggested"
            reason = reason_by_key[key]
            sg = sugg_by_key[key]
            on_apply = lambda s=sg: self._apply_suggestion(s)
        dot.set_state(state, reason, on_apply)

    def _resolve_sibling(self, filename) -> str | None:
        model = self.model_edit.text()
        mounts = self.mounts_panel.mounts()
        for d in (posixpath.dirname(model), posixpath.dirname(posixpath.dirname(model))):
            container = posixpath.join(d, filename)
            host = container_to_host(container, mounts)
            if host and os.path.exists(host):
                return container
        return None

    def _apply_suggestion(self, sg) -> None:
        for key, val in sg.settings.items():
            if key in self._widgets:
                self._widgets[key].set_value(val)
        for field, filename in sg.fields.items():
            container = self._resolve_sibling(filename)
            if container is None:
                continue
            if field == "mmproj":
                self.mmproj_edit.setText(container)
            elif field == "draft_model":
                self.draft_model_edit.setText(container)
        self.refresh_preview()
        self.apply_model_caps()

    @staticmethod
    def _meta_caps_text(meta, size, caps) -> str:
        bits = []
        if size:
            bits.append(f"{size / 1024**3:.1f} GiB")
        if meta and meta.quant:
            bits.append(meta.quant)
        if meta and meta.size_label:
            bits.append(meta.size_label)
        if caps:
            if caps.is_moe:
                bits.append(f"MoE {caps.expert_count}")
            if caps.has_mtp:
                bits.append("MTP" + (" (file)" if caps.mtp_sibling else ""))
            if caps.has_vision:
                bits.append("vision")
            if caps.has_swa:
                bits.append("SWA")
            if caps.ctx_train:
                bits.append(f"{caps.ctx_train // 1024}K ctx")
        return "  ·  ".join(bits)
