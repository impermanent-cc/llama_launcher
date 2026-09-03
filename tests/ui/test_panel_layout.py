from llama_launcher.ui.panels.lora_panel import LoraPanel
from llama_launcher.ui.panels.mounts_panel import MountsPanel


def test_mounts_host_container_columns_start_wide(qtbot):
    # Interactive (user-resizable) with the path columns given the lion's
    # share of the initial width.
    p = MountsPanel()
    qtbot.addWidget(p)
    assert p.table.columnWidth(0) >= 150  # Host
    assert p.table.columnWidth(1) >= 100  # Container


def test_mounts_table_has_a_multi_row_floor(qtbot):
    # Folders must keep 2-4 rows visible, not collapse to ~1 when the Environment
    # form is crowded with the native-launch rows.
    p = MountsPanel()
    qtbot.addWidget(p)
    assert p.table.minimumHeight() >= 140


def test_cards_strip_fits_a_full_card(qtbot):
    # One row of StatCards must show at full height (title/health/tok-s/KV), not
    # clip when the log view claims the vertical space.
    from llama_launcher.ui.panels.monitor_panel import MonitorPanel
    from llama_launcher.ui.widgets.stat_card import StatCard

    mp = MonitorPanel()
    qtbot.addWidget(mp)
    assert mp._cards_scroll.minimumHeight() >= StatCard("x").sizeHint().height()


def test_lora_path_column_starts_wide(qtbot):
    p = LoraPanel()
    qtbot.addWidget(p)
    assert p.table.columnWidth(0) >= 200  # Path


def test_new_flags_and_speculative_group_present(qtbot):
    from PySide6.QtWidgets import QGroupBox

    from llama_launcher.ui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    for key in [
        "spec-type",
        "spec-draft-n-min",
        "cache-type-k-draft",
        "cache-type-v-draft",
        "no-mmproj-offload",
        "override-tensor",
        "swa-full",
        "context-shift",
        "ctx-checkpoints",
        "checkpoint-min-step",
        "dry-sequence-breaker",
        "numa",
        "threads-http",
        "no-webui",
        "reasoning-format",
    ]:
        assert key in w._configure_panel._widgets, key
    titles = {b.title() for b in w.findChildren(QGroupBox)}
    assert "Speculative Decoding" in titles


def test_ik_engine_extends_spec_type_enum(qtbot):
    from llama_launcher.core.settings_catalog import CATALOG
    from llama_launcher.ui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    panel = w._configure_panel
    combo = panel.engine_combo
    combo.setCurrentIndex(combo.findData("ik_llama.cpp"))
    spec = panel._widgets["spec-type"]
    spec.set_value("suffix")
    assert spec.value() == "suffix"
    # flipping back to mainline drops the ik-only choice to the default
    combo.setCurrentIndex(combo.findData("llama.cpp"))
    assert spec.value() == CATALOG["spec-type"].default


def test_build_tab_present(qtbot):
    from llama_launcher.ui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    labels = [w.tabs.tabText(i) for i in range(w.tabs.count())]
    assert "Build" in labels
