import pytest

from llama_launcher.ui.main_window import MainWindow
from llama_launcher.ui.panels.mounts_panel import MountsPanel


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
