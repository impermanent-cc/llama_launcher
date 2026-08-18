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


def test_new_flags_and_speculative_group_present(qtbot):
    from PySide6.QtWidgets import QGroupBox
    from llama_launcher.ui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    for key in ["spec-type", "spec-draft-n-min", "cache-type-k-draft",
                "cache-type-v-draft", "no-mmproj-offload", "override-tensor",
                "swa-full", "context-shift", "ctx-checkpoints",
                "checkpoint-min-step", "dry-sequence-breaker", "numa",
                "threads-http", "no-webui", "reasoning-format"]:
        assert key in w._configure_panel._widgets, key
    titles = {b.title() for b in w.findChildren(QGroupBox)}
    assert "Speculative Decoding" in titles
