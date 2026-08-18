from llama_launcher.ui.main_window import MainWindow


def test_window_has_three_tabs(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    titles = [w.tabs.tabText(i) for i in range(w.tabs.count())]
    assert titles == ["Configure", "Monitor", "Benchmark"]
    assert not hasattr(w, "router_panel")
    # preview + launch remain reachable (shared, below tabs)
    assert w._configure_panel.preview_text().startswith("podman run --rm")
    assert hasattr(w, "launch_btn")


def test_api_key_and_harness_on_configure(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    assert w._configure_panel.api_key_box is not None
    assert w._configure_panel.harness_box is not None


def test_models_table_on_monitor(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.router_models_table.table is not None
