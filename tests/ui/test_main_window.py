from llama_launcher.core.spec import Profile, Mount, Runtime
from llama_launcher.ui.main_window import MainWindow


def _profile():
    return Profile(
        name="UI Test", image="img:tag",
        runtime=Runtime(binary="podman", gpu_mode="cdi"),
        mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
        model="/models/m.gguf",
        settings={"temp": 0.6, "ctx-size": 4096, "port": 8080},
    )


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
