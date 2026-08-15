import pytest

from llama_launcher.ui.main_window import MainWindow
from llama_launcher.ui.panels.mounts_panel import MountsPanel
from llama_launcher.ui.panels.monitor_panel import MonitorPanel
from llama_launcher.ui.panels.benchmark_panel import BenchmarkPanel
from llama_launcher.ui.widgets.info_button import InfoButton


@pytest.fixture
def win(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def test_environment_fields_have_tooltips(win):
    for attr in ("image_edit", "model_edit", "binary_combo", "gpu_combo",
                 "mode_combo", "bind_host_combo", "mmproj_edit", "draft_model_edit",
                 "raw_edit", "extra_args_edit", "selinux_check"):
        w = getattr(win, attr)
        assert w.toolTip().strip(), f"{attr} is missing a tooltip"


def test_router_member_headers_have_tooltips(win):
    hdrs = [win.members_list.horizontalHeaderItem(c)
            for c in range(win.members_list.columnCount())]
    assert all(h is not None and h.toolTip().strip() for h in hdrs)


def test_folders_headers_have_tooltips(qtbot):
    panel = MountsPanel(); qtbot.addWidget(panel)
    hdrs = [panel.table.horizontalHeaderItem(c)
            for c in range(panel.table.columnCount())]
    assert all(h is not None and h.toolTip().strip() for h in hdrs)


def test_monitor_legend_is_behind_info_button(qtbot):
    p = MonitorPanel()
    qtbot.addWidget(p)
    assert not hasattr(p, "stats_legend")               # no always-on legend label
    btn = p.findChild(InfoButton)
    assert btn is not None and "gen" in btn.info_text    # legend text lives in the popover
    # The summary bar spans the whole width, so a tooltip on it fires on hover
    # anywhere along the bar -- duplicating the info button. Keep the legend
    # ONLY on the compact info button, not on the summary label.
    assert p.summary.toolTip() == ""


def test_benchmark_legend_is_behind_info_button(qtbot):
    p = BenchmarkPanel()
    qtbot.addWidget(p)
    assert not hasattr(p, "bench_legend")
    infos = p.findChildren(InfoButton)
    assert any("t/s" in b.info_text for b in infos)
    # header hover tips still applied
    assert p.bench_table.horizontalHeaderItem(2).toolTip() != ""
