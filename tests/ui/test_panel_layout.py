from PySide6.QtWidgets import QHeaderView

from llama_launcher.ui.panels.mounts_panel import MountsPanel
from llama_launcher.ui.panels.lora_panel import LoraPanel


def test_mounts_host_container_columns_stretch(qtbot):
    p = MountsPanel()
    qtbot.addWidget(p)
    hdr = p.table.horizontalHeader()
    assert hdr.sectionResizeMode(0) == QHeaderView.Stretch   # Host
    assert hdr.sectionResizeMode(1) == QHeaderView.Stretch   # Container


def test_lora_path_column_stretches(qtbot):
    p = LoraPanel()
    qtbot.addWidget(p)
    hdr = p.table.horizontalHeader()
    assert hdr.sectionResizeMode(0) == QHeaderView.Stretch   # Path
