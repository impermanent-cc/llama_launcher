from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit,
    QGroupBox, QScrollArea, QLabel, QPlainTextEdit, QPushButton, QComboBox,
    QMessageBox, QFileDialog, QInputDialog
)

from llama_launcher.core.spec import Profile, Mount, Runtime
from llama_launcher.core.settings_catalog import CATALOG
from llama_launcher.core.command_builder import build_command
from llama_launcher.core.validation import validate
from llama_launcher.store.profiles import (
    default_base_dir, list_profiles, save_profile, delete_profile,
)
from llama_launcher.services import runtime, terminal, registry, health
from llama_launcher.services.registry import split_image, variant_prefix
from llama_launcher.ui.widgets.setting_widgets import make_widget
from llama_launcher.ui.panels.mounts_panel import MountsPanel


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

        body = QHBoxLayout()
        root.addLayout(body)

        # LEFT: environment (image + model only for v1 binding; mounts editor TODO-UI)
        left = QGroupBox("Environment")
        left_form = QFormLayout(left)
        self.image_edit = QLineEdit()
        self.model_edit = QLineEdit()
        self.binary_combo = QComboBox(); self.binary_combo.addItems(["podman", "docker"])
        self.gpu_combo = QComboBox(); self.gpu_combo.addItems(["cdi", "gpus-all", "none"])
        for w in (self.image_edit, self.model_edit):
            w.textChanged.connect(self.refresh_preview)
        self.binary_combo.currentTextChanged.connect(self.refresh_preview)
        self.gpu_combo.currentTextChanged.connect(self.refresh_preview)
        left_form.addRow("Image", self.image_edit)
        left_form.addRow("Model", self.model_edit)
        left_form.addRow("Runtime", self.binary_combo)
        left_form.addRow("GPU", self.gpu_combo)

        # mounts editor
        self.mounts_panel = MountsPanel()
        self.mounts_panel.changed.connect(self.refresh_preview)
        left_form.addRow("Folders", self.mounts_panel)
        self.mmproj_edit = QLineEdit(); self.mmproj_edit.textChanged.connect(self.refresh_preview)
        self.raw_edit = QLineEdit(); self.raw_edit.textChanged.connect(self.refresh_preview)
        left_form.addRow("mmproj", self.mmproj_edit)
        left_form.addRow("Raw args", self.raw_edit)
        self.fetch_btn = QPushButton("Fetch latest build")
        self.fetch_btn.clicked.connect(self.on_fetch_latest)
        left_form.addRow(self.fetch_btn)

        body.addWidget(left, 1)

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

        # BOTTOM: preview + buttons
        self.preview = QPlainTextEdit(); self.preview.setReadOnly(True)
        root.addWidget(QLabel("Command preview:"))
        root.addWidget(self.preview)
        buttons = QHBoxLayout()
        self.launch_btn = QPushButton("▶ Launch")
        self.stop_btn = QPushButton("■ Stop")
        self.restart_btn = QPushButton("⟳ Restart")
        for b in (self.launch_btn, self.stop_btn, self.restart_btn):
            buttons.addWidget(b)
        root.addLayout(buttons)

        # profile bar (added to the top of root via insertLayout)
        bar = QHBoxLayout()
        self.profile_combo = QComboBox()
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

    def load_profile(self, p: Profile) -> None:
        self._profile = p
        self.image_edit.setText(p.image)
        self.model_edit.setText(p.model)
        self.binary_combo.setCurrentText(p.runtime.binary)
        self.gpu_combo.setCurrentText(p.runtime.gpu_mode)
        self.mounts_panel.set_mounts(p.mounts)
        self.mmproj_edit.setText(p.mmproj or "")
        self.raw_edit.setText(p.raw_args)
        for key, w in self._widgets.items():
            w.set_value(w.setting.default)
            if key in p.settings:
                w.set_value(p.settings[key])
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
                            gpu_mode=self.gpu_combo.currentText(),
                            selinux_label_disable=self._profile.runtime.selinux_label_disable,
                            extra_run_args=self._profile.runtime.extra_run_args),
            mounts=self.mounts_panel.mounts(),
            model=self.model_edit.text(),
            mmproj=self.mmproj_edit.text() or None,
            loras=list(self._profile.loras),
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

    def on_launch(self):
        if not self._validate_or_warn():
            return
        argv = build_command(self.current_profile())
        terminal.launch(argv)

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
        name = f"llama-{slugify(self._profile.name)}"
        state = runtime.container_state(name, p.runtime.binary)
        ok = health.health_ok(p.settings.get("port", 8080)) if state == "running" else False
        self.status_label.setText("● " + health.derive_status(state, ok))
