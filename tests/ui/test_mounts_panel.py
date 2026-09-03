from llama_launcher.core.spec import Mount
from llama_launcher.ui.panels.mounts_panel import MountsPanel


def test_mounts_panel_roundtrip(qtbot):
    panel = MountsPanel()
    qtbot.addWidget(panel)
    mounts = [
        Mount(
            host="/h/models", container="/models", role="model", mode="ro", selinux="z"
        ),
        Mount(
            host="/h/ws",
            container="/workspace",
            role="workspace",
            mode="rw",
            workdir=True,
        ),
    ]
    panel.set_mounts(mounts)
    out = panel.mounts()
    assert out[0].container == "/models" and out[0].mode == "ro"
    assert out[0].selinux == "z"
    assert out[1].workdir is True and out[1].mode == "rw"
    assert out[1].selinux is None


def test_mounts_panel_item_changed_emits_signal(qtbot):
    panel = MountsPanel()
    qtbot.addWidget(panel)
    panel.set_mounts(
        [Mount(host="/h/models", container="/models", role="model", mode="ro")]
    )
    with qtbot.waitSignal(panel.changed, timeout=1000):
        panel.table.item(0, 0).setText("/new")


def test_add_row_does_not_crash_when_changed_reads_mounts(qtbot):
    # itemChanged fires mid-construction; a slot that
    # calls mounts() must not crash on a half-built row (cellWidget -> None).
    panel = MountsPanel()
    qtbot.addWidget(panel)
    panel.changed.connect(panel.mounts)  # simulates main window reading mounts
    panel._add_row(Mount(host="/h", container="/c"))
    out = panel.mounts()
    assert len(out) == 1
    assert out[0].host == "/h" and out[0].container == "/c"


def test_role_combo_change_emits_zero_arg_changed(qtbot):
    # combo currentTextChanged passes an arg; the 0-arg
    # changed signal must still fire without a TypeError.
    panel = MountsPanel()
    qtbot.addWidget(panel)
    panel.set_mounts([Mount(host="/h", container="/c", role="model", mode="ro")])
    count = {"n": 0}
    panel.changed.connect(lambda: count.__setitem__("n", count["n"] + 1))
    panel.table.cellWidget(0, 2).setCurrentText("workspace")
    assert count["n"] >= 1
    assert panel.mounts()[0].role == "workspace"
