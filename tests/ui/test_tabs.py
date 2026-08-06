from llama_launcher.ui.main_window import MainWindow


def test_window_has_three_tabs(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    titles = [w.tabs.tabText(i) for i in range(w.tabs.count())]
    assert titles == ["Configure", "Monitor", "Router"]
    # preview + launch remain reachable (shared, below tabs)
    assert w.preview_text().startswith("podman run --rm")
    assert hasattr(w, "launch_btn")
