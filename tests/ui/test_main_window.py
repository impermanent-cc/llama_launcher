from llama_launcher.core.spec import Profile, Mount, Runtime, LoraRef
from llama_launcher.ui.main_window import MainWindow


def _profile():
    return Profile(
        name="UI Test", image="img:tag",
        runtime=Runtime(binary="podman", gpu_mode="cdi"),
        mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
        model="/models/m.gguf",
        settings={"temp": 0.6, "ctx-size": 4096, "port": 8080},
    )


def test_configure_tab_body_and_bottom_are_a_vertical_splitter(qtbot):
    """The Configure tab's two-column body and its command-preview/api-key
    section sit in a vertical QSplitter so the user can drag the divider to
    give the Environment/Settings columns more or less height."""
    from PySide6.QtWidgets import QSplitter
    from PySide6.QtCore import Qt
    w = MainWindow()
    qtbot.addWidget(w)
    sp = w._configure_splitter
    assert isinstance(sp, QSplitter)
    assert sp.orientation() == Qt.Vertical
    assert sp.count() == 2
    # the command-preview / api-key / harness block is the draggable bottom pane
    assert sp.widget(1) is w._config_bottom


def test_name_field_drives_profile_name_and_container(qtbot):
    """Typing in the Name field sets the profile name (and thus --name/container)."""
    w = MainWindow()
    qtbot.addWidget(w)
    w.name_edit.setText("My Cool Model")
    assert w.current_profile().name == "My Cool Model"
    assert w._container_name() == "llama-my-cool-model"
    assert "--name llama-my-cool-model" in w.preview_text()


def test_load_profile_populates_name_field(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.load_profile(_profile())          # name="UI Test"
    assert w.name_edit.text() == "UI Test"


def test_window_constructs(qtbot):
    from llama_launcher.core.settings_catalog import CATALOG
    w = MainWindow()
    qtbot.addWidget(w)
    # the settings grid built a widget for every catalog entry
    assert set(w._widgets.keys()) == set(CATALOG.keys())
    assert w.preview_text().startswith("podman run --rm")


def test_load_profile_and_preview(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.load_profile(_profile())
    text = w.preview_text()
    assert text.startswith("podman run --rm --name llama-ui-test")
    assert "--temp 0.6" in text
    assert "--ctx-size 4096" in text
    assert "--host 0.0.0.0" in text


def test_roundtrip_profile(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    p = _profile()
    w.load_profile(p)
    out = w.current_profile()
    assert out.model == "/models/m.gguf"
    assert out.settings.get("ctx-size") == 4096
    assert out.image == "img:tag"


def test_loras_roundtrip(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    p = _profile()
    p.loras = [LoraRef(path="/models/lora.gguf", scale=0.5)]
    w.load_profile(p)
    out = w.current_profile()
    assert len(out.loras) == 1
    assert out.loras[0].path == "/models/lora.gguf"
    assert abs(out.loras[0].scale - 0.5) < 1e-6


def test_advanced_podman_settings_roundtrip(qtbot):
    """load_profile -> current_profile round-trips extra_run_args and selinux_label_disable,
    and the preview contains the expected flags."""
    w = MainWindow()
    qtbot.addWidget(w)
    p = _profile()
    p.runtime = Runtime(
        binary="podman",
        gpu_mode="cdi",
        extra_run_args="--cap-add SYS_NICE",
        selinux_label_disable=True,
    )
    w.load_profile(p)
    out = w.current_profile()
    assert out.runtime.extra_run_args == "--cap-add SYS_NICE"
    assert out.runtime.selinux_label_disable is True
    preview = w.preview_text()
    assert "--cap-add SYS_NICE" in preview
    assert "--security-opt=label=disable" in preview
