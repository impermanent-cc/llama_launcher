from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit,
    QGroupBox, QScrollArea, QLabel, QPlainTextEdit, QPushButton, QComboBox
)

from llama_launcher.core.spec import Profile, Mount, Runtime
from llama_launcher.core.settings_catalog import CATALOG
from llama_launcher.core.command_builder import build_command
from llama_launcher.ui.widgets.setting_widgets import make_widget


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

        self.refresh_preview()

    def load_profile(self, p: Profile) -> None:
        self._profile = p
        self.image_edit.setText(p.image)
        self.model_edit.setText(p.model)
        self.binary_combo.setCurrentText(p.runtime.binary)
        self.gpu_combo.setCurrentText(p.runtime.gpu_mode)
        for key, w in self._widgets.items():
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
            mounts=list(self._profile.mounts),
            model=self.model_edit.text(),
            mmproj=self._profile.mmproj,
            loras=list(self._profile.loras),
            settings=settings,
            raw_args=self._profile.raw_args,
        )

    def preview_text(self) -> str:
        return " ".join(build_command(self.current_profile()))

    def refresh_preview(self) -> None:
        self.preview.setPlainText(self.preview_text())
