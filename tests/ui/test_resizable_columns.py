"""Every data column in the app's tables is user-resizable (Interactive).

Stretch and ResizeToContents both lock the column against dragging; only
fixed-size button/checkbox columns may keep ResizeToContents.
"""
from PySide6.QtWidgets import QHeaderView


def _assert_interactive(table, cols):
    hdr = table.horizontalHeader()
    for c in cols:
        assert hdr.sectionResizeMode(c) == QHeaderView.Interactive, \
            f"column {c} not user-resizable"


def test_mounts_columns_resizable(main_window):
    _assert_interactive(main_window._configure_panel.mounts_panel.table, range(6))


def test_members_columns_resizable(main_window):
    _assert_interactive(main_window._configure_panel.members_list, range(4))


def test_rpc_workers_columns_resizable(main_window):
    _assert_interactive(main_window._configure_panel.rpc_workers_table.table, range(4))


def test_lora_columns_resizable(main_window):
    # Path + Scale; the trailing Browse-button column stays sized to content.
    _assert_interactive(main_window._configure_panel.lora_panel.table, (0, 1))


def test_router_models_columns_resizable(main_window):
    # Model id + Status; the trailing buttons column stays sized to content.
    _assert_interactive(main_window.router_models_table.table, (0, 1))
