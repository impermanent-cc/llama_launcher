from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QGridLayout, QCheckBox, QLabel, QLineEdit, QToolButton
)

from llama_launcher.core.settings_catalog import Setting
from llama_launcher.ui.widgets.no_wheel import (
    NoWheelComboBox, NoWheelSpinBox, NoWheelDoubleSpinBox,
)


class SuggestionDot(QToolButton):
    """Inline per-setting indicator: filled ● = suggested, hollow ○ = N/A.

    When a concrete value suggestion exists, the dot is clickable and applies
    it (on_apply); otherwise it is a passive indicator. Hover explains why.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoRaise(True)
        self._on_apply = None
        self.clicked.connect(self._fire)
        self.set_state("none")

    def set_state(self, state: str, reason: str = "", on_apply=None) -> None:
        self._on_apply = on_apply
        self.setToolTip(reason)
        if state == "suggested":
            self.setText("●")
            self.setStyleSheet("QToolButton { color: palette(highlight); border: none; }")
            self.setVisible(True)
        elif state == "muted":
            self.setText("○")
            self.setStyleSheet("QToolButton { color: palette(mid); border: none; }")
            self.setVisible(True)
        else:  # none
            self.setText("")
            self.setVisible(False)
        self.setEnabled(on_apply is not None)

    def _fire(self) -> None:
        if self._on_apply is not None:
            self._on_apply()


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
            # Start at the catalog default. A no-op for the runtime catalog
            # (every bool there defaults False) but load-bearing for the build
            # catalog's default-ON CMake options: an unchecked box for those
            # would read as "explicitly OFF" and emit -DNAME=OFF.
            self._editor.setChecked(bool(setting.default))
            self._editor.toggled.connect(lambda: self.changed.emit())
        elif t == "enum":
            self._editor = NoWheelComboBox()
            self._editor.addItems(list(setting.enum))
            self._editor.setCurrentText(str(setting.default))
            self._editor.currentTextChanged.connect(lambda: self.changed.emit())
        elif t == "int" and setting.suggestions:
            # editable combo of suggested values; any integer can still be typed
            self._editor = NoWheelComboBox()
            self._editor.setEditable(True)
            self._editor.addItems([str(s) for s in setting.suggestions])
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
            grid = QGridLayout(container)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(2)
            help_map = dict(setting.option_help)
            self._all_check = QCheckBox("all")
            self._all_check.setToolTip(tooltip)   # overall (danger) note on "all"
            self._all_check.toggled.connect(self._on_all_toggled)
            self._all_check.toggled.connect(lambda: self.changed.emit())
            boxes = [self._all_check]
            for opt in setting.enum:
                cb = QCheckBox(opt)
                cb.setToolTip(help_map.get(opt, tooltip))   # per-option description
                cb.toggled.connect(lambda: self.changed.emit())
                self._checks[opt] = cb
                boxes.append(cb)
            # arrange in a compact grid (3 columns) so nothing overflows horizontally
            for i, cb in enumerate(boxes):
                grid.addWidget(cb, i // 3, i % 3)
            self._editor = container
        else:  # string
            self._editor = QLineEdit()
            # Start at the catalog default, same as every other branch: a
            # fresh editor for a non-empty-default setting (cors-origins "*",
            # blas-vendor "Generic", ...) must not read as "explicitly set to
            # blank" -- that polluted saved configs with phantom ""-valued
            # options and made is_set() True on untouched forms.
            if setting.default:
                self._editor.setText(str(setting.default))
            if getattr(setting, "secret", False):
                # Mask the value on screen (shoulder-surfing / screenshots) with a
                # reveal toggle; the value is still readable programmatically.
                self._editor.setEchoMode(QLineEdit.Password)
                self._editor.setToolTip((self._editor.toolTip() + "  ").strip())
                reveal = QToolButton()
                reveal.setText("👁")
                reveal.setCheckable(True)
                reveal.setToolTip("Show / hide")
                reveal.toggled.connect(
                    lambda on: self._editor.setEchoMode(
                        QLineEdit.Normal if on else QLineEdit.Password))
                self._reveal_btn = reveal
            self._editor.textChanged.connect(lambda: self.changed.emit())

        # Cap editor widths so dropdowns/inputs don't stretch the whole panel,
        # and left-align them with a trailing stretch.
        _max_width = {"enum": 150, "int_or_token": 150, "int": 120,
                      "float": 120, "string": 240}.get(t)
        if t == "int" and setting.suggestions:
            _max_width = 150   # editable preset combo needs room for 6-digit values
        if _max_width:
            self._editor.setMaximumWidth(_max_width)
        layout.addWidget(self._editor)
        if getattr(self, "_reveal_btn", None) is not None:
            layout.addWidget(self._reveal_btn)
        layout.addStretch(1)

        self._dot = SuggestionDot(self)
        layout.addWidget(self._dot)

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
            if self.setting.suggestions:
                try:
                    return int(self._editor.currentText().strip())
                except ValueError:
                    return self.setting.default
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
        elif t == "int" and self.setting.suggestions:
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

    def set_enum_choices(self, choices) -> None:
        """Replace an enum editor's items (engine-dependent value sets), keeping
        the current selection if it survives, else the setting default. No-op
        for non-enum widgets."""
        if self.setting.type != "enum":
            return
        items = [str(c) for c in choices]
        cur = self._editor.currentText()
        self._editor.blockSignals(True)
        self._editor.clear()
        self._editor.addItems(items)
        self._editor.setCurrentText(cur if cur in items else str(self.setting.default))
        self._editor.blockSignals(False)

    def is_set(self) -> bool:
        return self.value() != self.setting.default

    def set_suggestion(self, state: str, reason: str = "", on_apply=None) -> None:
        self._dot.set_state(state, reason, on_apply)


def make_widget(setting: Setting) -> SettingWidget:
    return SettingWidget(setting)


def make_row_label(setting: Setting) -> QLabel:
    """Form-row label for a setting: the flag name, plus a muted '*deprecated'
    marker for flags upstream is retiring (kept working for older images). The
    tooltip points at the replacement so users know where to go."""
    label = QLabel(setting.flag)
    if setting.deprecated:
        label.setText(
            f"{setting.flag} "
            "<span style='color: palette(mid); font-size: 90%;'>*deprecated</span>"
        )
        label.setToolTip(
            "Deprecated upstream in favor of --load-mode; still works on older "
            "images. See this row's control for details."
        )
    return label
