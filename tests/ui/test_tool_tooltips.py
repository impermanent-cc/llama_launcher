from llama_launcher.core.settings_catalog import CATALOG
from llama_launcher.ui.widgets.setting_widgets import make_widget


def test_each_tool_checkbox_has_distinct_tooltip(qtbot):
    w = make_widget(CATALOG["tools"])
    qtbot.addWidget(w)
    tips = {opt: cb.toolTip() for opt, cb in w._checks.items()}
    # every tool has a non-empty tooltip...
    assert all(tips.values())
    # ...and they are no longer all identical (the reported bug)
    assert len(set(tips.values())) == len(tips)
    # spot-check a few are actually about the right tool
    assert "read" in tips["read_file"].lower()
    assert "DANGER" in tips["exec_shell_command"]
    assert "date" in tips["get_datetime"].lower()
