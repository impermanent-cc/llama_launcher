from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QCheckBox, QLineEdit
)

from llama_launcher.core.settings_catalog import Setting
from llama_launcher.ui.widgets.no_wheel import (
    NoWheelComboBox, NoWheelSpinBox, NoWheelDoubleSpinBox,
)


class SettingWidget(QWidget):
    changed = Signal()

    def __init__(self, setting: Setting, parent=None):
        super().__init__(parent)
        self.setting = setting
        self._editor = None
        self._all_check = None
        self._checks: dict[str, QCheckBox] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        t = setting.type

        tooltip = setting.tooltip
        if setting.danger:
            tooltip = "⚠ DANGER: " + tooltip
            self.setObjectName("dangerSetting")

        if t == "bool":
            self._editor = QCheckBox(setting.flag)
            self._editor.toggled.connect(lambda: self.changed.emit())
        elif t == "enum":
            self._editor = NoWheelComboBox()
            self._editor.addItems(list(setting.enum))
            self._editor.setCurrentText(str(setting.default))
            self._editor.currentTextChanged.connect(lambda: self.changed.emit())
        elif t == "int":
            self._editor = NoWheelSpinBox()
            self._editor.setRange(int(setting.minimum if setting.minimum is not None else -2**31),
                                  int(setting.maximum if setting.maximum is not None else 2**31 - 1))
            self._editor.setSingleStep(int(setting.step or 1))
            self._editor.setValue(int(setting.default))
            self._editor.valueChanged.connect(lambda: self.changed.emit())
        elif t == "float":
            self._editor = NoWheelDoubleSpinBox()
            self._editor.setDecimals(3)
            self._editor.setRange(float(setting.minimum if setting.minimum is not None else -1e9),
                                  float(setting.maximum if setting.maximum is not None else 1e9))
            self._editor.setSingleStep(float(setting.step or 0.01))
            self._editor.setValue(float(setting.default))
            self._editor.valueChanged.connect(lambda: self.changed.emit())
        elif t == "int_or_token":
            self._editor = NoWheelComboBox()
            self._editor.setEditable(True)
            self._editor.addItems(list(setting.tokens))
            self._editor.setCurrentText(str(setting.default))
            self._editor.currentTextChanged.connect(lambda: self.changed.emit())
        elif t == "multiselect":
            container = QWidget()
            row = QHBoxLayout(container)
            row.setContentsMargins(0, 0, 0, 0)
            self._all_check = QCheckBox("all")
            self._all_check.toggled.connect(self._on_all_toggled)
            self._all_check.toggled.connect(lambda: self.changed.emit())
            row.addWidget(self._all_check)
            for opt in setting.enum:
                cb = QCheckBox(opt)
                cb.toggled.connect(lambda: self.changed.emit())
                self._checks[opt] = cb
                row.addWidget(cb)
            self._editor = container
            if setting.danger:
                for cb in [self._all_check, *self._checks.values()]:
                    cb.setToolTip(tooltip)
        else:  # string
            self._editor = QLineEdit()
            self._editor.textChanged.connect(lambda: self.changed.emit())

        layout.addWidget(self._editor)

        if setting.danger:
            self.setStyleSheet("#dangerSetting { border: 1px solid red; }")
        self.setToolTip(tooltip)
        self._editor.setToolTip(tooltip)

    def _on_all_toggled(self, checked):
        for cb in self._checks.values():
            cb.setEnabled(not checked)

    def value(self):
        t = self.setting.type
        if t == "bool":
            return self._editor.isChecked()
        if t == "enum":
            return self._editor.currentText()
        if t == "int":
            return self._editor.value()
        if t == "float":
            return self._editor.value()
        if t == "int_or_token":
            text = self._editor.currentText().strip()
            if text == "":
                return self.setting.default
            if text in self.setting.tokens:
                return text
            try:
                return int(text)
            except ValueError:
                return text
        if t == "multiselect":
            if self._all_check.isChecked():
                return "all"
            return ",".join(opt for opt, cb in self._checks.items() if cb.isChecked())
        return self._editor.text()

    def set_value(self, v):
        t = self.setting.type
        if t == "bool":
            self._editor.setChecked(bool(v))
        elif t == "enum":
            self._editor.setCurrentText(str(v))
        elif t in ("int", "float"):
            self._editor.setValue(v)
        elif t == "int_or_token":
            self._editor.setCurrentText(str(v))
        elif t == "multiselect":
            if v == "all":
                self._all_check.setChecked(True)
            else:
                self._all_check.setChecked(False)
                members = set(str(v).split(",")) if v else set()
                for opt, cb in self._checks.items():
                    cb.setChecked(opt in members)
        else:
            self._editor.setText(str(v))

    def is_set(self) -> bool:
        return self.value() != self.setting.default


def make_widget(setting: Setting) -> SettingWidget:
    return SettingWidget(setting)
