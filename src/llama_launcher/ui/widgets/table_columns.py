"""Shared column-sizing policy for the app's QTableWidgets.

Every data column is user-resizable (Interactive): Stretch and
ResizeToContents both lock a column against dragging. Only fixed-size
button/checkbox columns may keep ResizeToContents. Centralized so the next
policy change (or the next new table) touches one place.
"""

from PySide6.QtWidgets import QHeaderView


def set_resizable_columns(table, widths, content_cols=()) -> None:
    """Make `table`'s columns user-resizable with sensible initial widths.

    widths: initial pixel width per column, positionally; use None to leave a
    column at its default width. content_cols: indices of fixed button-style
    columns that size themselves to content instead.
    """
    hdr = table.horizontalHeader()
    hdr.setSectionResizeMode(QHeaderView.Interactive)
    for col in content_cols:
        hdr.setSectionResizeMode(col, QHeaderView.ResizeToContents)
    for col, w in enumerate(widths):
        if w is not None and col not in content_cols:
            table.setColumnWidth(col, w)
