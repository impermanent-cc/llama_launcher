"""The app must apply its own SVG as the window/tray icon (not the desktop
default 'yellow w'). The icon lives at assets/llama-launcher.svg and is wired
via llama_launcher.ui.icon.app_icon()."""


def test_find_repo_icon_locates_asset():
    from llama_launcher.ui.icon import _find_repo_icon
    p = _find_repo_icon()
    assert p is not None, "could not locate assets/llama-launcher.svg from the package"
    assert p.name == "llama-launcher.svg"
    assert p.is_file()


def test_app_icon_is_non_null(qtbot):
    """QIcon must actually load the SVG (requires the qsvg plugin); a null icon
    would mean the window still shows the default."""
    from llama_launcher.ui.icon import app_icon
    ic = app_icon()
    assert not ic.isNull()
    # SVG icons are scalable (availableSizes() is empty), so prove it actually
    # renders by asking for a concrete pixmap; null/empty would mean the qsvg
    # engine failed and the window would still show the default icon.
    pm = ic.pixmap(64, 64)
    assert not pm.isNull(), "icon produced a null 64px pixmap; SVG failed to render"
    assert pm.width() > 0 and pm.height() > 0


def test_main_window_sets_window_icon(qtbot):
    from llama_launcher.ui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    assert not w.windowIcon().isNull(), "MainWindow did not set its window icon"
