from llama_launcher.core.spec import Profile, Mount, Runtime
from llama_launcher.ui.main_window import MainWindow


def test_gpu_dropdown_labels_map_to_canonical_values(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    # labels are human-readable, but stored data stays canonical
    datas = [w._configure_panel.gpu_combo.itemData(i) for i in range(w._configure_panel.gpu_combo.count())]
    assert datas == ["cdi", "gpus-all", "none"]
    labels = [w._configure_panel.gpu_combo.itemText(i) for i in range(w._configure_panel.gpu_combo.count())]
    assert any("nvidia.com/gpu=all" in t for t in labels)


def test_selecting_cdi_emits_device_flag(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.gpu_combo.setCurrentIndex(0)  # CDI
    assert "--device nvidia.com/gpu=all" in w._configure_panel.preview_text()
    assert w._configure_panel.current_profile().runtime.gpu_mode == "cdi"


def test_load_profile_selects_gpu_by_value(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    p = Profile(name="g", image="img", runtime=Runtime(gpu_mode="gpus-all"),
                mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
                model="/models/m.gguf", settings={"port": 8080})
    w._configure_panel.load_profile(p)
    assert w._configure_panel.gpu_combo.currentData() == "gpus-all"
    assert "--gpus all" in w._configure_panel.preview_text()
    assert w._configure_panel.current_profile().runtime.gpu_mode == "gpus-all"
