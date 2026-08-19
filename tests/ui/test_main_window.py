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


def test_name_field_drives_profile_name_and_container(qtbot):
    """Typing in the Name field sets the profile name (and thus --name/container)."""
    w = MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.name_edit.setText("My Cool Model")
    assert w._configure_panel.current_profile().name == "My Cool Model"
    assert w._container_name() == "llama-my-cool-model"
    assert "--name llama-my-cool-model" in w._configure_panel.preview_text()


def test_load_profile_populates_name_field(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(_profile())          # name="UI Test"
    assert w._configure_panel.name_edit.text() == "UI Test"


def test_window_constructs(qtbot):
    from llama_launcher.core.settings_catalog import CATALOG
    w = MainWindow()
    qtbot.addWidget(w)
    # the settings grid built a widget for every catalog entry
    assert set(w._configure_panel._widgets.keys()) == set(CATALOG.keys())
    assert w._configure_panel.preview_text().startswith("podman run --rm")


def test_load_profile_and_preview(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(_profile())
    text = w._configure_panel.preview_text()
    assert text.startswith("podman run --rm --name llama-ui-test")
    assert "--temp 0.6" in text
    assert "--ctx-size 4096" in text
    assert "--host 0.0.0.0" in text


def test_roundtrip_profile(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    p = _profile()
    w._configure_panel.load_profile(p)
    out = w._configure_panel.current_profile()
    assert out.model == "/models/m.gguf"
    assert out.settings.get("ctx-size") == 4096
    assert out.image == "img:tag"


def test_loras_roundtrip(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    p = _profile()
    p.loras = [LoraRef(path="/models/lora.gguf", scale=0.5)]
    w._configure_panel.load_profile(p)
    out = w._configure_panel.current_profile()
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
    w._configure_panel.load_profile(p)
    out = w._configure_panel.current_profile()
    assert out.runtime.extra_run_args == "--cap-add SYS_NICE"
    assert out.runtime.selinux_label_disable is True
    preview = w._configure_panel.preview_text()
    assert "--cap-add SYS_NICE" in preview
    assert "--security-opt=label=disable" in preview


def test_stop_timeout_roundtrips_through_profile(qtbot):
    """The Stop grace-period spinbox round-trips load_profile -> current_profile,
    so an edited grace period is actually saved onto the profile."""
    w = MainWindow()
    qtbot.addWidget(w)
    p = Profile(name="slow", runtime=Runtime(stop_timeout=45))
    w._configure_panel.load_profile(p)
    assert w._configure_panel.stop_timeout_spin.value() == 45
    assert w._configure_panel.current_profile().runtime.stop_timeout == 45


def test_engine_roundtrips_through_profile(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    p = Profile(name="ik", runtime=Runtime(engine="ik_llama.cpp"))
    w._configure_panel.load_profile(p)
    assert w._configure_panel.engine_combo.currentData() == "ik_llama.cpp"
    assert w._configure_panel.current_profile().runtime.engine == "ik_llama.cpp"


def test_ik_flags_hidden_on_llama_cpp_engine(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(Profile(name="m", runtime=Runtime(engine="llama.cpp")))
    assert "run-time-repack" not in w._configure_panel.active_catalog()
    w._configure_panel.load_profile(Profile(name="ik", runtime=Runtime(engine="ik_llama.cpp")))
    assert "run-time-repack" in w._configure_panel.active_catalog()


def test_switch_to_ik_seeds_default_image_when_empty(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.image_edit.setText("")
    w._configure_panel.engine_combo.setCurrentIndex(w._configure_panel.engine_combo.findData("ik_llama.cpp"))
    assert w._configure_panel.image_edit.text() == "ghcr.io/ikawrakow/ik-llama-cpp:cu12-server"


def test_switch_engine_does_not_clobber_user_image(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.image_edit.setText("myregistry.local/custom:tag")
    w._configure_panel.engine_combo.setCurrentIndex(w._configure_panel.engine_combo.findData("ik_llama.cpp"))
    assert w._configure_panel.image_edit.text() == "myregistry.local/custom:tag"


def test_ik_cache_type_enum_gains_extras(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.engine_combo.setCurrentIndex(w._configure_panel.engine_combo.findData("ik_llama.cpp"))
    ctk = w._configure_panel._widgets["cache-type-k"]
    ctk.set_value("q6_0")
    assert ctk.value() == "q6_0"


def test_launch_mode_round_trips_through_profile(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    p = Profile(name="n", runtime=Runtime(launch_mode="native",
                native_binary="/opt/bin/llama-server"))
    w._configure_panel.load_profile(p)
    assert w._configure_panel.launch_mode_combo.currentData() == "native"
    assert w._configure_panel.native_binary_edit.text() == "/opt/bin/llama-server"
    out = w._configure_panel.current_profile()
    assert out.runtime.launch_mode == "native"
    assert out.runtime.native_binary == "/opt/bin/llama-server"


def test_native_mode_hides_container_fields(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(Profile(name="n",
        runtime=Runtime(launch_mode="native", native_binary="/x")))
    cp = w._configure_panel
    assert cp.native_binary_edit.isVisibleTo(cp)
    assert not cp.image_edit.isVisibleTo(cp)          # Image hidden
    # detached_check is reparented into MainWindow's button row, so isVisibleTo(cp)
    # is trivially False; assert against its real parent so this checks the actual
    # setVisible() state set by _update_detached_visibility (always-managed-bg native).
    assert not cp.detached_check.isVisibleTo(cp.detached_check.parentWidget())
    assert not cp.extra_args_edit.isVisibleTo(cp)     # "Extra podman args" hidden
    assert not cp.selinux_check.isVisibleTo(cp)       # SELinux checkbox hidden


def test_container_mode_shows_container_fields(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._configure_panel.load_profile(Profile(name="c", image="img",
        runtime=Runtime(launch_mode="container")))
    cp = w._configure_panel
    assert cp.image_edit.isVisibleTo(cp)
    assert not cp.native_binary_edit.isVisibleTo(cp)
    assert cp.extra_args_edit.isVisibleTo(cp)
    # detached checkbox is shown for a container server profile (see native test)
    assert cp.detached_check.isVisibleTo(cp.detached_check.parentWidget())
    assert cp.selinux_check.isVisibleTo(cp)
