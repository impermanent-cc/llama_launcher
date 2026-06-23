import os
import subprocess
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit,
    QGroupBox, QScrollArea, QLabel, QPlainTextEdit, QPushButton,
    QMessageBox, QFileDialog, QInputDialog, QTabWidget
)

from llama_launcher.core.spec import Profile, Mount, Runtime
from llama_launcher.core.settings_catalog import CATALOG
from llama_launcher.core.command_builder import build_command
from llama_launcher.core.pathmap import host_to_container
from llama_launcher.core.validation import validate
from llama_launcher.store.profiles import (
    default_base_dir, list_profiles, save_profile, delete_profile,
)
from llama_launcher.services import runtime, terminal, registry, health, metrics, gpu, model_info
from llama_launcher.core import vram
from llama_launcher.services.registry import split_image, variant_prefix
from llama_launcher.ui.widgets.setting_widgets import make_widget
from llama_launcher.ui.widgets.no_wheel import NoWheelComboBox
from llama_launcher.ui.panels.mounts_panel import MountsPanel
from llama_launcher.ui.panels.lora_panel import LoraPanel
from llama_launcher.ui.widgets.collapsible import CollapsibleSection
from llama_launcher.ui.panels.monitor_panel import MonitorPanel


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

        # LEFT: environment (image + model only for v1 binding; mounts editor TODO-UI)
        left = QGroupBox("Environment")
        left_form = QFormLayout(left)
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
        self.model_edit.textChanged.connect(
            lambda _: self.model_meta_label.setText(self.model_meta_text())
        )
        self.binary_combo.currentTextChanged.connect(self.refresh_preview)
        self.gpu_combo.currentTextChanged.connect(self.refresh_preview)
        left_form.addRow("Image", self.image_edit)
        left_form.addRow("Model", self._field_with_browse(self.model_edit))
        left_form.addRow("Runtime", self.binary_combo)
        left_form.addRow("GPU", self.gpu_combo)

        # mounts editor
        self.mounts_panel = MountsPanel()
        self.mounts_panel.changed.connect(self.refresh_preview)
        left_form.addRow("Folders", self.mounts_panel)
        self.mmproj_edit = QLineEdit(); self.mmproj_edit.textChanged.connect(self.refresh_preview)
        self.raw_edit = QLineEdit(); self.raw_edit.textChanged.connect(self.refresh_preview)
        left_form.addRow("mmproj", self._field_with_browse(self.mmproj_edit))
        self.lora_panel = LoraPanel()
        self.lora_panel.changed.connect(self.refresh_preview)
        self.lora_section = CollapsibleSection("LoRA adapters", self.lora_panel, collapsed=True)
        left_form.addRow(self.lora_section)
        left_form.addRow("Raw args", self.raw_edit)
        self.fetch_btn = QPushButton("Fetch latest build")
        self.fetch_btn.clicked.connect(self.on_fetch_latest)
        left_form.addRow(self.fetch_btn)

        body.addWidget(left, 3)

        # RIGHT: settings grouped, scrollable
        self._widgets: dict[str, object] = {}
        right_scroll = QScrollArea(); right_scroll.setWidgetResizable(True)
        right_inner = QWidget(); right_layout = QVBoxLayout(right_inner)
        groups: dict[str, QFormLayout] = {}
        for key, setting in CATALOG.items():
            if setting.group not in groups:
                box = QGroupBox(setting.group)
                groups[setting.group] = QFormLayout(box)
                right_layout.addWidget(box)
            w = make_widget(setting)
            w.changed.connect(self.refresh_preview)
            self._widgets[key] = w
            groups[setting.group].addRow(setting.flag, w)
        right_scroll.setWidget(right_inner)
        body.addWidget(right_scroll, 2)
        self.configure_tab = configure_tab

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(configure_tab, "Configure")
        self.monitor_panel = MonitorPanel()
        self.tabs.addTab(self.monitor_panel, "Monitor")
        root.addWidget(self.tabs)

        # BOTTOM: preview + buttons (shared, below both tabs)
        self.model_meta_label = QLabel("")
        root.addWidget(self.model_meta_label)
        preview_row = QHBoxLayout()
        self.preview = QPlainTextEdit(); self.preview.setReadOnly(True)
        preview_row.addWidget(self.preview, 1)
        self.export_sh_btn = QPushButton("Export .sh")
        self.export_sh_btn.clicked.connect(self._on_export_sh)
        preview_row.addWidget(self.export_sh_btn)
        root.addWidget(QLabel("Command preview:"))
        root.addLayout(preview_row)
        buttons = QHBoxLayout()
        self.launch_btn = QPushButton("▶ Launch")
        self.stop_btn = QPushButton("■ Stop")
        self.restart_btn = QPushButton("⟳ Restart")
        self.web_ui_btn = QPushButton("Open Web UI")
        self.web_ui_btn.setEnabled(False)
        self.web_ui_btn.clicked.connect(self.open_web_ui)
        for b in (self.launch_btn, self.stop_btn, self.restart_btn, self.web_ui_btn):
            buttons.addWidget(b)
        root.addLayout(buttons)

        # profile bar (added to the top of root via insertLayout)
        bar = QHBoxLayout()
        self.profile_combo = NoWheelComboBox()
        self.save_btn = QPushButton("Save")
        self.save_as_btn = QPushButton("Save As")
        self.delete_btn = QPushButton("Delete")
        self.status_label = QLabel("● stopped")
        bar.addWidget(self.profile_combo, 1)
        for b in (self.save_btn, self.save_as_btn, self.delete_btn):
            bar.addWidget(b)
        bar.addWidget(self.status_label)
        root.insertLayout(0, bar)
        self.save_btn.clicked.connect(self.save_current_profile)
        self.save_as_btn.clicked.connect(self.save_as_profile)
        self.delete_btn.clicked.connect(self.delete_current_profile)
        self.profile_combo.activated.connect(self._on_pick_profile)

        # lifecycle buttons
        self.launch_btn.clicked.connect(self.on_launch)
        self.stop_btn.clicked.connect(self.on_stop)
        self.restart_btn.clicked.connect(self.on_restart)

        self._reload_profile_list()

        self.refresh_preview()

        self._log_proc = None

        from PySide6.QtWidgets import QSystemTrayIcon, QMenu, QStyle
        self._really_quit = False
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        self.tray.setToolTip("Llama Launcher")
        menu = QMenu()
        menu.addAction("Show", self.showNormal)
        menu.addAction("Launch", self.on_launch)
        menu.addAction("Stop", self.on_stop)
        menu.addSeparator()
        menu.addAction("Quit", self.quit_app)
        self.tray.setContextMenu(menu)
        self.tray.show()

        self.lora_panel.set_browse_resolver(
            lambda h: host_to_container(h, self.mounts_panel.mounts())
        )

        from PySide6.QtCore import QTimer
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(2000)
        self._status_timer.timeout.connect(self.update_status)
        self._status_timer.start()

    def _field_with_browse(self, line_edit: QLineEdit) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(line_edit, 1)
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
        self._profile = p
        self.image_edit.setText(p.image)
        self.model_edit.setText(p.model)
        self.binary_combo.setCurrentText(p.runtime.binary)
        self.gpu_combo.setCurrentIndex(max(0, self.gpu_combo.findData(p.runtime.gpu_mode)))
        self.mounts_panel.set_mounts(p.mounts)
        self.mmproj_edit.setText(p.mmproj or "")
        self.lora_panel.set_loras(p.loras)
        self.raw_edit.setText(p.raw_args)
        for key, w in self._widgets.items():
            w.set_value(w.setting.default)
            if key in p.settings:
                w.set_value(p.settings[key])
        self.model_meta_label.setText(self.model_meta_text())
        self.refresh_preview()

    def current_profile(self) -> Profile:
        settings = {}
        for key, w in self._widgets.items():
            if w.is_set():
                settings[key] = w.value()
        # port is always stored
        settings["port"] = self._widgets["port"].value()
        return Profile(
            name=self._profile.name,
            image=self.image_edit.text(),
            runtime=Runtime(binary=self.binary_combo.currentText(),
                            gpu_mode=self.gpu_combo.currentData(),
                            selinux_label_disable=self._profile.runtime.selinux_label_disable,
                            extra_run_args=self._profile.runtime.extra_run_args),
            mounts=self.mounts_panel.mounts(),
            model=self.model_edit.text(),
            mmproj=self.mmproj_edit.text() or None,
            loras=self.lora_panel.loras(),
            settings=settings,
            raw_args=self.raw_edit.text(),
        )

    def preview_text(self) -> str:
        return " ".join(build_command(self.current_profile()))

    def refresh_preview(self) -> None:
        self.preview.setPlainText(self.preview_text())

    def _reload_profile_list(self):
        self.profile_combo.clear()
        self._profiles = {p.name: p for p in list_profiles(base_dir())}
        self.profile_combo.addItems(list(self._profiles.keys()))

    def _on_pick_profile(self, _index):
        name = self.profile_combo.currentText()
        if name in self._profiles:
            self.load_profile(self._profiles[name])

    def save_current_profile(self):
        p = self.current_profile()
        p.name = self._profile.name
        save_profile(p, base_dir())
        self._reload_profile_list()
        self.profile_combo.setCurrentText(p.name)

    def save_as_profile(self):
        name, ok = QInputDialog.getText(self, "Save As", "Profile name:")
        if ok and name:
            self._profile.name = name
            self.save_current_profile()

    def delete_current_profile(self):
        name = self.profile_combo.currentText()
        if name:
            delete_profile(name, base_dir())
            self._reload_profile_list()

    def _validate_or_warn(self) -> bool:
        p = self.current_profile()
        issues = validate(p, binary_found=runtime.binary_available(p.runtime.binary))
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
        meta = model_info.read_gguf_meta(p.model) if p.model else None
        free = gpu.free_vram_bytes()
        if meta is None or free is None or not meta.n_layers or not meta.n_embd:
            return None
        ctx = p.settings.get("ctx-size") or meta.ctx_train or 4096
        if ctx == 0:
            ctx = meta.ctx_train or 4096
        est = vram.estimate(
            n_layers=meta.n_layers, n_head=meta.n_head or 1,
            n_head_kv=meta.n_head_kv or meta.n_head or 1, n_embd=meta.n_embd, ctx=ctx,
            k_quant=p.settings.get("cache-type-k", "f16"),
            v_quant=p.settings.get("cache-type-v", "f16"),
            weights_bytes=model_info.file_size(p.model) or 0,
        )
        ok, margin = vram.fits(est.total_bytes, free)
        if ok:
            return None
        gib = 1024 ** 3
        return (f"Estimated VRAM need ~{est.total_bytes/gib:.1f} GiB exceeds free "
                f"~{free/gib:.1f} GiB by ~{-margin/gib:.1f} GiB. It may not fit — "
                f"consider quantized KV cache (-ctk/-ctv q8_0) or a higher --n-cpu-moe. "
                f"(Estimate is conservative; --n-cpu-moe/-ngl reduce actual GPU use.)")

    def on_launch(self):
        if not self._validate_or_warn():
            return
        warn = self.vram_check()
        if warn:
            QMessageBox.warning(self, "VRAM check", warn)
        argv = build_command(self.current_profile())
        terminal.launch(argv)
        self._start_log_follower()

    def on_stop(self):
        from llama_launcher.core.spec import slugify
        p = self.current_profile()
        runtime.stop(f"llama-{slugify(self._profile.name)}", p.runtime.binary)

    def on_restart(self):
        self.on_stop()
        self.on_launch()

    def on_fetch_latest(self):
        repo, tag = split_image(self.image_edit.text())
        if not repo:
            return
        prefix = variant_prefix(tag) if tag else "server-cuda12"
        latest = registry.fetch_latest(repo, prefix)
        if latest:
            self.image_edit.setText(f"{repo}:{latest}")

    def update_status(self):
        from llama_launcher.core.spec import slugify
        p = self.current_profile()
        if not runtime.binary_available(p.runtime.binary):
            self.status_label.setText("● stopped")
            self.web_ui_btn.setEnabled(False)
            return
        name = f"llama-{slugify(self._profile.name)}"
        state = runtime.container_state(name, p.runtime.binary)
        ok = health.health_ok(p.settings.get("port", 8080)) if state == "running" else False
        self.status_label.setText("● " + health.derive_status(state, ok))
        self.web_ui_btn.setEnabled(state == "running")
        if state == "running":
            self.monitor_panel.update_stats(self.collect_monitor_data())

    def collect_monitor_data(self) -> dict:
        from llama_launcher.core.spec import slugify
        from llama_launcher.services.metrics import kv_usage_ratio
        p = self.current_profile()
        port = p.settings.get("port", 8080)
        metrics_on = bool(p.settings.get("metrics"))
        m = metrics.fetch_metrics(port) if metrics_on else {}
        slots = metrics.fetch_slots(port)
        name = f"llama-{slugify(self._profile.name)}"
        st = runtime.stats(name, p.runtime.binary) or {}
        return {
            "tok_s": m.get("llamacpp:predicted_tokens_seconds"),
            "prompt_tok_s": m.get("llamacpp:prompt_tokens_seconds"),
            "kv_pct": kv_usage_ratio(slots),
            "gpus": gpu.query_gpus(),
            "cpu": st.get("cpu_perc", ""),
            "mem": st.get("mem_usage", ""),
            "uptime": "",
            "metrics_on": metrics_on,
        }

    def _start_log_follower(self):
        from PySide6.QtCore import QProcess
        from llama_launcher.core.spec import slugify
        self._stop_log_follower()
        p = self.current_profile()
        name = f"llama-{slugify(self._profile.name)}"
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(
            lambda: self.monitor_panel.append_log(bytes(proc.readAllStandardOutput()).decode("utf-8", "replace")))
        argv = runtime.logs_argv(name, p.runtime.binary)
        proc.start(argv[0], argv[1:])
        self._log_proc = proc

    def _stop_log_follower(self):
        if self._log_proc is not None:
            self._log_proc.kill()
            self._log_proc = None

    def _on_export_sh(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export shell script", "run.sh",
                                              "Shell scripts (*.sh);;All files (*)")
        if path:
            self.export_sh(path)

    def open_web_ui(self):
        port = self.current_profile().settings.get("port", 8080)
        subprocess.Popen(["xdg-open", f"http://127.0.0.1:{port}"], start_new_session=True)

    def export_sh(self, path: str):
        cmd = " ".join(build_command(self.current_profile()))
        Path(path).write_text(f"#!/usr/bin/env bash\n{cmd}\n")
        os.chmod(path, 0o755)

    def closeEvent(self, event):
        if getattr(self, "_really_quit", False):
            self._stop_log_follower()
            event.accept()
        else:
            event.ignore()
            self.hide()

    def quit_app(self):
        self._really_quit = True
        from PySide6.QtWidgets import QApplication
        QApplication.instance().quit()

    def model_meta_text(self) -> str:
        p = self.current_profile()
        if not p.model:
            return ""
        meta = model_info.read_gguf_meta(p.model)
        size = model_info.file_size(p.model)
        bits = []
        if size:
            bits.append(f"{size / 1024**3:.1f} GiB")
        if meta and meta.quant:
            bits.append(meta.quant)
        if meta and meta.size_label:
            bits.append(meta.size_label)
        return "  ·  ".join(bits)
