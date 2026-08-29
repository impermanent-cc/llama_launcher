import datetime
from llama_launcher.core.settings_catalog import for_engine
from llama_launcher.core.build_catalog import BUILD_CATALOG


def _panel(qtbot, tmp_path):
    from llama_launcher.ui.panels.build_panel import BuildPanel
    p = BuildPanel(base_dir=tmp_path)
    qtbot.addWidget(p)
    return p


def test_form_engine_gated_both_ways(qtbot, tmp_path):
    p = _panel(qtbot, tmp_path)
    p.engine_combo.setCurrentIndex(p.engine_combo.findData("llama.cpp"))
    assert "cpu-repack" in p._widgets and "iqk-fa-all-quants" not in p._widgets
    p.engine_combo.setCurrentIndex(p.engine_combo.findData("ik_llama.cpp"))
    assert "iqk-fa-all-quants" in p._widgets and "cpu-repack" not in p._widgets


def test_native_preview_contains_cmake_pair(qtbot, tmp_path):
    p = _panel(qtbot, tmp_path)
    p.target_combo.setCurrentIndex(p.target_combo.findData("native"))
    p._widgets["cuda"].set_value(True)
    p.refresh_preview()
    text = p.preview.toPlainText()
    assert "cmake -B build-" in text and "-DGGML_CUDA=ON" in text
    assert "cmake --build" in text


def test_generate_container_writes_containerfile_and_registry(qtbot, tmp_path, monkeypatch):
    from llama_launcher.store.builds import load_outputs
    p = _panel(qtbot, tmp_path)
    p.target_combo.setCurrentIndex(p.target_combo.findData("container"))
    p.name_edit.setText("srv")
    p.generate()
    outs = load_outputs(tmp_path)
    assert len(outs) == 1 and outs[0].kind == "tag"
    assert outs[0].identifier.endswith(datetime.date.today().strftime("%Y%m%d"))
    assert (tmp_path / "builds" / "srv.containerfile").exists()


def test_cuda_arch_prefill_never_clobbers(qtbot, tmp_path, monkeypatch):
    import llama_launcher.ui.panels.build_panel as bp
    monkeypatch.setattr(bp, "query_compute_caps", lambda: ["120"])
    p = _panel(qtbot, tmp_path)
    p._widgets["cuda"].set_value(True)
    assert p._widgets["cuda-architectures"].value() == "120"
    p._widgets["cuda-architectures"].set_value("86")
    p._widgets["cuda"].set_value(False)
    p._widgets["cuda"].set_value(True)
    assert p._widgets["cuda-architectures"].value() == "86"


def test_outputs_table_statuses(qtbot, tmp_path, monkeypatch):
    import llama_launcher.ui.panels.build_panel as bp
    from llama_launcher.store.builds import add_output
    from llama_launcher.core.build_spec import BuildOutput
    add_output(BuildOutput(id="a1", kind="tag", identifier="llama-custom:x-1",
                           config_name="x", engine="llama.cpp", git_ref="m",
                           options={}, created="2026-08-28"), tmp_path)
    monkeypatch.setattr(bp, "list_images_detailed", lambda *a, **k: {})
    p = _panel(qtbot, tmp_path)
    p.refresh_outputs_sync()      # test hook: same logic, no thread pool
    statuses = [p.outputs_table.item(r, 1).text()
                for r in range(p.outputs_table.rowCount())]
    assert statuses == ["missing"]


def test_delete_refused_when_profile_uses_tag(qtbot, tmp_path, monkeypatch):
    import llama_launcher.ui.panels.build_panel as bp
    from llama_launcher.store.builds import add_output
    from llama_launcher.store.profiles import save_profile
    from llama_launcher.core.build_spec import BuildOutput
    from llama_launcher.core.spec import Profile
    from llama_launcher.services.runtime import ImageInfo

    add_output(BuildOutput(id="t1", kind="tag", identifier="t:1",
                           config_name="x", engine="llama.cpp", git_ref="m",
                           options={}, created="2026-08-28"), tmp_path)
    save_profile(Profile(name="p1", image="t:1"), tmp_path)

    monkeypatch.setattr(
        bp, "list_images_detailed",
        lambda *a, **k: {"t:1": ImageInfo(tag="t:1", size="10MB", created="now")})

    def _raise(*a, **k):
        raise AssertionError("remove_image should not be called")
    monkeypatch.setattr(bp, "remove_image", _raise)

    p = _panel(qtbot, tmp_path)
    p.refresh_outputs_sync()

    errors = []
    monkeypatch.setattr(p, "_error", lambda text: errors.append(text))

    p.outputs_table.setCurrentCell(0, 0)
    p.delete_selected_output()

    assert errors and "p1" in errors[0]


def test_delete_built_tag_confirms_then_removes(qtbot, tmp_path, monkeypatch):
    import llama_launcher.ui.panels.build_panel as bp
    from llama_launcher.store.builds import add_output, load_outputs
    from llama_launcher.core.build_spec import BuildOutput
    from llama_launcher.services.runtime import ImageInfo

    add_output(BuildOutput(id="t1", kind="tag", identifier="t:1",
                           config_name="x", engine="llama.cpp", git_ref="m",
                           options={}, created="2026-08-28"), tmp_path)

    monkeypatch.setattr(
        bp, "list_images_detailed",
        lambda *a, **k: {"t:1": ImageInfo(tag="t:1", size="10MB", created="now")})

    calls = []

    def _remove_image(binary, tag, connection=""):
        calls.append((binary, tag))
        return (True, "")
    monkeypatch.setattr(bp, "remove_image", _remove_image)

    p = _panel(qtbot, tmp_path)
    p.refresh_outputs_sync()
    monkeypatch.setattr(p, "_confirm", lambda text: True)

    p.outputs_table.setCurrentCell(0, 0)
    p.delete_selected_output()

    assert calls and calls[0][1] == "t:1"
    assert load_outputs(tmp_path) == []


def test_delete_binary_refuses_non_build_dir(qtbot, tmp_path, monkeypatch):
    import llama_launcher.ui.panels.build_panel as bp
    from llama_launcher.store.builds import add_output, load_outputs
    from llama_launcher.core.build_spec import BuildOutput

    add_output(BuildOutput(id="b1", kind="binary", identifier="/usr/bin/llama-server",
                           config_name="x", engine="llama.cpp", git_ref="m",
                           options={}, created="2026-08-28"), tmp_path)

    monkeypatch.setattr(bp, "list_images_detailed", lambda *a, **k: {})

    p = _panel(qtbot, tmp_path)
    # Force this binary's status to "built" regardless of whether the path
    # actually exists on the test machine -- the point of this test is the
    # rmtree safety guard, not binary_exists's own logic.
    monkeypatch.setattr(p, "_binary_exists", lambda path: True)
    p.refresh_outputs_sync()

    errors = []
    monkeypatch.setattr(p, "_confirm", lambda text: True)
    monkeypatch.setattr(p, "_error", lambda text: errors.append(text))

    p.outputs_table.setCurrentCell(0, 0)
    p.delete_selected_output()

    assert errors
    assert len(load_outputs(tmp_path)) == 1


def test_eligible_profiles_filtered_by_kind(qtbot, tmp_path, monkeypatch):
    import llama_launcher.ui.panels.build_panel as bp
    from llama_launcher.core.spec import Profile, Runtime
    from llama_launcher.store.profiles import save_profile
    from PySide6.QtCore import Qt
    save_profile(Profile(name="cont", image="x:1"), tmp_path)
    save_profile(Profile(name="nat", runtime=Runtime(launch_mode="native")), tmp_path)
    monkeypatch.setattr(bp, "list_images_detailed", lambda *a, **k: {})
    p = _panel(qtbot, tmp_path)
    assert p._eligible_profiles("tag") == ["cont"]
    assert p._eligible_profiles("binary") == ["nat"]
    assert p.outputs_table.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu


def test_maybe_seed_reseeds_generated_values_but_not_user_typed(qtbot, tmp_path):
    # Container target seeds the debian (non-CUDA) pair first.
    p = _panel(qtbot, tmp_path)
    p.target_combo.setCurrentIndex(p.target_combo.findData("container"))
    assert p.builder_image_edit.text() == "docker.io/library/debian:bookworm"
    assert p.runtime_image_edit.text() == "docker.io/library/debian:bookworm-slim"

    # Ticking cuda AFTER picking container must re-seed to the CUDA pair --
    # the fields still hold generator output, not user-typed text.
    p._widgets["cuda"].set_value(True)
    assert p.builder_image_edit.text() == \
        "docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04"
    assert p.runtime_image_edit.text() == \
        "docker.io/nvidia/cuda:12.8.1-runtime-ubuntu24.04"

    # A genuinely user-typed builder value must survive further cuda toggles.
    p.builder_image_edit.setText("my/custom:builder")
    p._widgets["cuda"].set_value(False)
    p._widgets["cuda"].set_value(True)
    assert p.builder_image_edit.text() == "my/custom:builder"
    # The untouched runtime field keeps tracking the generator's output.
    assert p.runtime_image_edit.text() == \
        "docker.io/nvidia/cuda:12.8.1-runtime-ubuntu24.04"


def test_refresh_preview_matches_generate_tag_on_collision(qtbot, tmp_path):
    import datetime
    from llama_launcher.core.build_command import auto_tag
    from llama_launcher.store.builds import load_outputs

    p = _panel(qtbot, tmp_path)
    p.target_combo.setCurrentIndex(p.target_combo.findData("container"))
    p.name_edit.setText("srv")
    p.generate()          # first generate: records the base tag

    cfg = p.current_build_config()
    existing = {o.identifier for o in load_outputs(tmp_path)}
    expected_tag = auto_tag(cfg, existing, datetime.date.today())
    assert expected_tag.endswith("-2")   # sanity: a collision really occurred

    p.refresh_preview()
    podman_line = p.preview.toPlainText().splitlines()[-1]
    assert expected_tag in podman_line


def test_delete_built_calls_pooled_refresh_not_sync(qtbot, tmp_path, monkeypatch):
    import llama_launcher.ui.panels.build_panel as bp
    from llama_launcher.store.builds import add_output
    from llama_launcher.core.build_spec import BuildOutput
    from llama_launcher.services.runtime import ImageInfo

    add_output(BuildOutput(id="t1", kind="tag", identifier="t:1",
                           config_name="x", engine="llama.cpp", git_ref="m",
                           options={}, created="2026-08-28"), tmp_path)
    monkeypatch.setattr(
        bp, "list_images_detailed",
        lambda *a, **k: {"t:1": ImageInfo(tag="t:1", size="10MB", created="now")})
    monkeypatch.setattr(bp, "remove_image", lambda *a, **k: (True, ""))

    p = _panel(qtbot, tmp_path)
    p.refresh_outputs_sync()
    monkeypatch.setattr(p, "_confirm", lambda text: True)

    pooled_calls, sync_calls = [], []
    monkeypatch.setattr(p, "refresh_outputs", lambda *a, **k: pooled_calls.append(1))
    monkeypatch.setattr(p, "refresh_outputs_sync", lambda *a, **k: sync_calls.append(1))

    p.outputs_table.setCurrentCell(0, 0)
    p.delete_selected_output()

    assert pooled_calls == [1]
    assert sync_calls == []


def test_delete_missing_calls_pooled_refresh_not_sync(qtbot, tmp_path, monkeypatch):
    import llama_launcher.ui.panels.build_panel as bp
    from llama_launcher.store.builds import add_output
    from llama_launcher.core.build_spec import BuildOutput

    add_output(BuildOutput(id="m1", kind="tag", identifier="t:missing",
                           config_name="x", engine="llama.cpp", git_ref="m",
                           options={}, created="2026-08-28"), tmp_path)
    monkeypatch.setattr(bp, "list_images_detailed", lambda *a, **k: {})

    p = _panel(qtbot, tmp_path)
    p.refresh_outputs_sync()
    monkeypatch.setattr(p, "_confirm", lambda text: True)

    pooled_calls, sync_calls = [], []
    monkeypatch.setattr(p, "refresh_outputs", lambda *a, **k: pooled_calls.append(1))
    monkeypatch.setattr(p, "refresh_outputs_sync", lambda *a, **k: sync_calls.append(1))

    p.outputs_table.setCurrentCell(0, 0)
    p.delete_selected_output()

    assert pooled_calls == [1]
    assert sync_calls == []


def test_use_in_profile_sets_image(qtbot, tmp_path, monkeypatch):
    import llama_launcher.ui.panels.build_panel as bp
    from llama_launcher.core.spec import Profile
    from llama_launcher.store.profiles import save_profile, list_profiles
    from llama_launcher.store.builds import add_output
    from llama_launcher.core.build_spec import BuildOutput
    save_profile(Profile(name="serv", image="old:1"), tmp_path)
    add_output(BuildOutput(id="a1", kind="tag", identifier="llama-custom:new-1",
                           config_name="x", engine="llama.cpp", git_ref="m",
                           options={}, created="2026-08-28"), tmp_path)
    monkeypatch.setattr(bp, "list_images_detailed", lambda *a, **k: {})
    p = _panel(qtbot, tmp_path)
    p.refresh_outputs_sync()
    p.outputs_table.selectRow(0)
    p.use_in_profile("serv")
    assert [pr.image for pr in list_profiles(tmp_path)] == ["llama-custom:new-1"]


def test_preview_and_outputs_share_tabbed_splitter(qtbot, tmp_path):
    from PySide6.QtWidgets import QSplitter, QTabWidget
    p = _panel(qtbot, tmp_path)
    splitters = p.findChildren(QSplitter)
    assert len(splitters) == 1
    tabs = p.bottom_tabs
    assert isinstance(tabs, QTabWidget)
    assert [tabs.tabText(i) for i in range(tabs.count())] == [
        "Command preview", "Outputs"]
    assert tabs.widget(0).isAncestorOf(p.preview)
    assert tabs.widget(1).isAncestorOf(p.outputs_table)
    assert splitters[0].isAncestorOf(tabs)


def test_fresh_panel_emits_only_touched_options(qtbot, tmp_path):
    # Regression net for the default-init class of bug: a fresh panel with just
    # CUDA ticked must not emit explicit OFFs or empty values for anything else.
    p = _panel(qtbot, tmp_path)
    p.target_combo.setCurrentIndex(p.target_combo.findData("container"))
    p.name_edit.setText("just-cuda")
    p._widgets["cuda"].set_value(True)
    p.refresh_preview()
    text = p.preview.toPlainText()
    assert "-DGGML_CUDA=ON" in text
    assert "=OFF" not in text
    assert "=''" not in text
