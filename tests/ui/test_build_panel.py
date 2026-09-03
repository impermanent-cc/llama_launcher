import datetime


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


def test_generate_container_writes_containerfile_and_registry(
    qtbot, tmp_path, monkeypatch
):
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
    # The nvidia-smi query runs off the UI thread (it can stall for seconds on
    # a wedged driver), so the first prefill lands asynchronously; afterwards
    # the result is cached and toggles are synchronous.
    import llama_launcher.ui.panels.build_panel as bp

    monkeypatch.setattr(bp, "query_compute_caps", lambda: ["120"])
    p = _panel(qtbot, tmp_path)
    p._widgets["cuda"].set_value(True)
    qtbot.waitUntil(
        lambda: p._widgets["cuda-architectures"].value() == "120", timeout=3000
    )
    p._widgets["cuda-architectures"].set_value("86")
    p._widgets["cuda"].set_value(False)
    p._widgets["cuda"].set_value(True)
    assert p._widgets["cuda-architectures"].value() == "86"


def test_cuda_prefill_not_queried_during_config_load(qtbot, tmp_path, monkeypatch):
    # Loading a saved cuda=ON config must not fire nvidia-smi (or reseed
    # anything): programmatic loads are not user gestures.
    import llama_launcher.ui.panels.build_panel as bp
    from llama_launcher.core.build_spec import BuildConfig

    calls = []
    monkeypatch.setattr(bp, "query_compute_caps", lambda: calls.append(1) or [])
    p = _panel(qtbot, tmp_path)
    p.load_build_config(BuildConfig(name="c", options={"cuda": True}))
    assert calls == []


def test_outputs_table_statuses(qtbot, tmp_path, monkeypatch):
    import llama_launcher.ui.panels.build_panel as bp
    from llama_launcher.core.build_spec import BuildOutput
    from llama_launcher.store.builds import add_output

    add_output(
        BuildOutput(
            id="a1",
            kind="tag",
            identifier="llama-custom:x-1",
            config_name="x",
            engine="llama.cpp",
            git_ref="m",
            options={},
            created="2026-08-28",
        ),
        tmp_path,
    )
    monkeypatch.setattr(bp, "list_images_detailed", lambda *a, **k: {})
    p = _panel(qtbot, tmp_path)
    p.refresh_outputs_sync()  # test hook: same logic, no thread pool
    statuses = [
        p.outputs_table.item(r, 1).text() for r in range(p.outputs_table.rowCount())
    ]
    assert statuses == ["missing"]


def test_delete_refused_when_profile_uses_tag(qtbot, tmp_path, monkeypatch):
    import llama_launcher.ui.panels.build_panel as bp
    from llama_launcher.core.build_spec import BuildOutput
    from llama_launcher.core.spec import Profile
    from llama_launcher.services.runtime import ImageInfo
    from llama_launcher.store.builds import add_output
    from llama_launcher.store.profiles import save_profile

    add_output(
        BuildOutput(
            id="t1",
            kind="tag",
            identifier="t:1",
            config_name="x",
            engine="llama.cpp",
            git_ref="m",
            options={},
            created="2026-08-28",
        ),
        tmp_path,
    )
    save_profile(Profile(name="p1", image="t:1"), tmp_path)

    monkeypatch.setattr(
        bp,
        "list_images_detailed",
        lambda *a, **k: {"t:1": ImageInfo(tag="t:1", size="10MB", created="now")},
    )

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
    from llama_launcher.core.build_spec import BuildOutput
    from llama_launcher.services.runtime import ImageInfo
    from llama_launcher.store.builds import add_output, load_outputs

    add_output(
        BuildOutput(
            id="t1",
            kind="tag",
            identifier="t:1",
            config_name="x",
            engine="llama.cpp",
            git_ref="m",
            options={},
            created="2026-08-28",
        ),
        tmp_path,
    )

    monkeypatch.setattr(
        bp,
        "list_images_detailed",
        lambda *a, **k: {"t:1": ImageInfo(tag="t:1", size="10MB", created="now")},
    )

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
    # rmi runs off-thread; wait for the poll chain to finish the delete.
    qtbot.waitUntil(lambda: load_outputs(tmp_path) == [], timeout=3000)

    assert calls and calls[0][1] == "t:1"


def test_delete_binary_refuses_non_build_dir(qtbot, tmp_path, monkeypatch):
    import llama_launcher.ui.panels.build_panel as bp
    from llama_launcher.core.build_spec import BuildOutput
    from llama_launcher.store.builds import add_output, load_outputs

    add_output(
        BuildOutput(
            id="b1",
            kind="binary",
            identifier="/usr/bin/llama-server",
            config_name="x",
            engine="llama.cpp",
            git_ref="m",
            options={},
            created="2026-08-28",
        ),
        tmp_path,
    )

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
    from PySide6.QtCore import Qt

    import llama_launcher.ui.panels.build_panel as bp
    from llama_launcher.core.spec import Profile, Runtime
    from llama_launcher.store.profiles import save_profile

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
    assert (
        p.builder_image_edit.text() == "docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04"
    )
    assert (
        p.runtime_image_edit.text()
        == "docker.io/nvidia/cuda:12.8.1-runtime-ubuntu24.04"
    )

    # A genuinely user-typed builder value must survive further cuda toggles.
    p.builder_image_edit.setText("my/custom:builder")
    p._widgets["cuda"].set_value(False)
    p._widgets["cuda"].set_value(True)
    assert p.builder_image_edit.text() == "my/custom:builder"
    # The untouched runtime field keeps tracking the generator's output.
    assert (
        p.runtime_image_edit.text()
        == "docker.io/nvidia/cuda:12.8.1-runtime-ubuntu24.04"
    )


def test_refresh_preview_matches_generate_tag_on_collision(qtbot, tmp_path):
    # A CHANGED config on the same day is a different expected build: it gets
    # the collision-bumped tag, and preview and registry must agree on it.
    from llama_launcher.store.builds import load_outputs

    p = _panel(qtbot, tmp_path)
    p.target_combo.setCurrentIndex(p.target_combo.findData("container"))
    p.name_edit.setText("srv")
    p.generate()  # first generate: records the base tag
    p._widgets["rpc"].set_value(True)  # changed flags -> new expected build

    p.refresh_preview()
    podman_line = p.preview.toPlainText().splitlines()[-1]
    assert "-2" in podman_line

    p.generate()
    outs = load_outputs(tmp_path)
    assert len(outs) == 2
    assert any(
        o.identifier.endswith("-2") and o.identifier in podman_line for o in outs
    )


def test_generate_twice_same_config_is_idempotent(qtbot, tmp_path):
    # Re-clicking Generate with nothing changed must not create phantom
    # "missing" rows: the identical expected build reuses its registry entry.
    from llama_launcher.store.builds import load_outputs

    p = _panel(qtbot, tmp_path)
    p.target_combo.setCurrentIndex(p.target_combo.findData("container"))
    p.name_edit.setText("srv")
    p.generate()
    p.generate()
    assert len(load_outputs(tmp_path)) == 1


def test_native_regenerate_replaces_entry(qtbot, tmp_path):
    # The same build dir can only hold one expected build: regenerating with
    # changed flags replaces the entry instead of stacking duplicates.
    from llama_launcher.store.builds import load_outputs

    p = _panel(qtbot, tmp_path)
    p.name_edit.setText("nat")
    p.source_dir_edit.setText("/s")
    p.generate()
    p.generate()
    assert len(load_outputs(tmp_path)) == 1
    p._widgets["rpc"].set_value(True)
    p.generate()
    outs = load_outputs(tmp_path)
    assert len(outs) == 1
    assert outs[0].options.get("rpc") is True


def test_load_config_does_not_reseed_loaded_images(qtbot, tmp_path):
    # A saved config that deliberately pairs debian bases with cuda=ON must
    # round-trip unmutated: loading is not a user gesture, so no reseeding.
    from llama_launcher.core.build_spec import BuildConfig

    p = _panel(qtbot, tmp_path)
    cfg = BuildConfig(
        name="deb-cuda",
        target="container",
        builder_image="docker.io/library/debian:bookworm",
        runtime_image="docker.io/library/debian:bookworm-slim",
        options={"cuda": True},
    )
    p.load_build_config(cfg)
    assert p.builder_image_edit.text() == "docker.io/library/debian:bookworm"
    assert p.current_build_config() == cfg


def test_engine_flip_preserves_shared_option_values(qtbot, tmp_path):
    p = _panel(qtbot, tmp_path)
    p._widgets["cuda"].set_value(True)  # engine="any"
    p._widgets["cuda-fa"].set_value(False)  # llama.cpp-only
    p.engine_combo.setCurrentIndex(p.engine_combo.findData("ik_llama.cpp"))
    assert p._widgets["cuda"].value() is True
    p.engine_combo.setCurrentIndex(p.engine_combo.findData("llama.cpp"))
    assert p._widgets["cuda"].value() is True
    assert p._widgets["cuda-fa"].value() is False


def test_load_config_forgets_prior_form_state(qtbot, tmp_path):
    # The engine-flip stash must not leak old form values into a freshly
    # loaded config on the next flip.
    from llama_launcher.core.build_spec import BuildConfig

    p = _panel(qtbot, tmp_path)
    p._widgets["cuda"].set_value(True)
    p.load_build_config(BuildConfig(name="clean"))
    p.engine_combo.setCurrentIndex(p.engine_combo.findData("ik_llama.cpp"))
    p.engine_combo.setCurrentIndex(p.engine_combo.findData("llama.cpp"))
    assert p._widgets["cuda"].value() is False


def test_use_in_profile_emits_profile_updated(qtbot, tmp_path, monkeypatch):
    import llama_launcher.ui.panels.build_panel as bp
    from llama_launcher.core.build_spec import BuildOutput
    from llama_launcher.core.spec import Profile
    from llama_launcher.store.builds import add_output
    from llama_launcher.store.profiles import save_profile

    save_profile(Profile(name="serv", image="old:1"), tmp_path)
    add_output(
        BuildOutput(
            id="a1",
            kind="tag",
            identifier="llama-custom:new-1",
            config_name="x",
            engine="llama.cpp",
            git_ref="m",
            options={},
            created="2026-08-28",
        ),
        tmp_path,
    )
    monkeypatch.setattr(bp, "list_images_detailed", lambda *a, **k: {})
    p = _panel(qtbot, tmp_path)
    p.refresh_outputs_sync()
    p.outputs_table.selectRow(0)
    with qtbot.waitSignal(p.profile_updated, timeout=1000) as blocker:
        p.use_in_profile("serv")
    assert blocker.args == ["serv"]


def test_delete_built_calls_pooled_refresh_not_sync(qtbot, tmp_path, monkeypatch):
    import llama_launcher.ui.panels.build_panel as bp
    from llama_launcher.core.build_spec import BuildOutput
    from llama_launcher.services.runtime import ImageInfo
    from llama_launcher.store.builds import add_output

    add_output(
        BuildOutput(
            id="t1",
            kind="tag",
            identifier="t:1",
            config_name="x",
            engine="llama.cpp",
            git_ref="m",
            options={},
            created="2026-08-28",
        ),
        tmp_path,
    )
    monkeypatch.setattr(
        bp,
        "list_images_detailed",
        lambda *a, **k: {"t:1": ImageInfo(tag="t:1", size="10MB", created="now")},
    )
    monkeypatch.setattr(bp, "remove_image", lambda *a, **k: (True, ""))

    p = _panel(qtbot, tmp_path)
    p.refresh_outputs_sync()
    monkeypatch.setattr(p, "_confirm", lambda text: True)

    pooled_calls, sync_calls = [], []
    monkeypatch.setattr(p, "refresh_outputs", lambda *a, **k: pooled_calls.append(1))
    monkeypatch.setattr(p, "refresh_outputs_sync", lambda *a, **k: sync_calls.append(1))

    p.outputs_table.setCurrentCell(0, 0)
    p.delete_selected_output()
    # rmi runs off-thread; the pooled refresh fires from the poll chain.
    qtbot.waitUntil(lambda: pooled_calls == [1], timeout=3000)

    assert sync_calls == []


def test_delete_missing_calls_pooled_refresh_not_sync(qtbot, tmp_path, monkeypatch):
    import llama_launcher.ui.panels.build_panel as bp
    from llama_launcher.core.build_spec import BuildOutput
    from llama_launcher.store.builds import add_output

    add_output(
        BuildOutput(
            id="m1",
            kind="tag",
            identifier="t:missing",
            config_name="x",
            engine="llama.cpp",
            git_ref="m",
            options={},
            created="2026-08-28",
        ),
        tmp_path,
    )
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
    from llama_launcher.core.build_spec import BuildOutput
    from llama_launcher.core.spec import Profile
    from llama_launcher.store.builds import add_output
    from llama_launcher.store.profiles import list_profiles, save_profile

    save_profile(Profile(name="serv", image="old:1"), tmp_path)
    add_output(
        BuildOutput(
            id="a1",
            kind="tag",
            identifier="llama-custom:new-1",
            config_name="x",
            engine="llama.cpp",
            git_ref="m",
            options={},
            created="2026-08-28",
        ),
        tmp_path,
    )
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
        "Command preview",
        "Outputs",
    ]
    assert tabs.widget(0).isAncestorOf(p.preview)
    assert tabs.widget(1).isAncestorOf(p.outputs_table)
    assert splitters[0].isAncestorOf(tabs)


def test_fresh_panel_emits_only_touched_options(qtbot, tmp_path):
    # A fresh panel with just CUDA ticked must not emit explicit OFFs or empty
    # values for anything else.
    p = _panel(qtbot, tmp_path)
    p.target_combo.setCurrentIndex(p.target_combo.findData("container"))
    p.name_edit.setText("just-cuda")
    p._widgets["cuda"].set_value(True)
    p.refresh_preview()
    text = p.preview.toPlainText()
    assert "-DGGML_CUDA=ON" in text
    assert "=OFF" not in text
    assert "=''" not in text


def test_fresh_form_has_no_phantom_options(qtbot, tmp_path):
    # Untouched string widgets with non-empty defaults (blas-vendor,
    # sycl-target) must not appear in options as "": a phantom "" entry
    # pollutes saved configs and breaks _matching_entry equality across load
    # round-trips.
    p = _panel(qtbot, tmp_path)
    assert p.current_build_config().options == {}


def test_cross_engine_load_does_not_leak_stash(qtbot, tmp_path):
    # Loading an ik config over a llama form must not re-stash the outgoing
    # llama-only values; otherwise flipping back to llama.cpp re-applies a
    # cuda-fa=False the loaded config never contained.
    from llama_launcher.core.build_spec import BuildConfig

    p = _panel(qtbot, tmp_path)
    p.engine_combo.setCurrentIndex(p.engine_combo.findData("llama.cpp"))
    p._widgets["cuda-fa"].set_value(False)  # llama-only, default True
    p.load_build_config(BuildConfig(name="ik", engine="ik_llama.cpp"))
    p.engine_combo.setCurrentIndex(p.engine_combo.findData("llama.cpp"))
    assert p._widgets["cuda-fa"].is_set() is False
    assert p._widgets["cuda-fa"].value() is True


def test_generate_binary_replaces_entry_on_ref_change(qtbot, tmp_path):
    # Same build dir + same options but a NEW git ref supersedes the entry:
    # the registry must not keep reporting the old ref as provenance.
    from llama_launcher.store.builds import load_outputs

    p = _panel(qtbot, tmp_path)
    p.target_combo.setCurrentIndex(p.target_combo.findData("native"))
    p.name_edit.setText("nat")
    p.source_dir_edit.setText("/s")
    p.ref_edit.setText("b6789")
    p.generate()
    p.ref_edit.setText("b7000")
    p.generate()
    outs = load_outputs(tmp_path)
    assert [o.git_ref for o in outs] == ["b7000"]


def test_overlapping_outputs_refresh_renders_newest(qtbot, tmp_path, monkeypatch):
    # Two refreshes in flight: only the NEWEST gather's result may render --
    # the first (stale) snapshot must be dropped, not painted last.
    import llama_launcher.ui.panels.build_panel as bp
    from llama_launcher.core.build_spec import BuildOutput
    from llama_launcher.services.runtime import ImageInfo
    from llama_launcher.store.builds import add_output

    add_output(
        BuildOutput(
            id="t1",
            kind="tag",
            identifier="t:1",
            config_name="x",
            engine="llama.cpp",
            git_ref="m",
            options={},
            created="2026-08-28",
        ),
        tmp_path,
    )
    p = _panel(qtbot, tmp_path)

    stale = {}  # first gather: image gone (pretend pre-delete snapshot)
    fresh = {"t:1": ImageInfo(tag="t:1", size="10MB", created="now")}
    results = iter([stale, fresh])
    monkeypatch.setattr(bp, "list_images_detailed", lambda *a, **k: next(results))

    # Synchronous pool: the real global pool gives no ordering guarantee
    # between the two gathers, so the iterator could feed 'stale' to the
    # NEWER gather. Running each gather inline pins gather 1 = stale and,
    # deliberately, has the stale gather FINISH FIRST: the exact case the
    # superseding logic must drop on render.
    class _InlinePool:
        @staticmethod
        def globalInstance():
            return _InlinePool()

        @staticmethod
        def start(runnable):
            runnable.run()

    monkeypatch.setattr(bp, "QThreadPool", _InlinePool)

    p.refresh_outputs()  # gather 1 (stale), completes immediately
    p.refresh_outputs()  # gather 2 (fresh) supersedes it
    qtbot.waitUntil(
        lambda: p._outputs_gather is None and p.outputs_table.rowCount() > 0,
        timeout=3000,
    )
    assert p.outputs_table.item(0, 1).text() == "built"


def test_delete_binary_uses_shared_build_dir_rule(qtbot, tmp_path, monkeypatch):
    # The rmtree target comes from the same extract_build_dir rule the in-use
    # guard uses; a real build-<slug> dir is deleted, others are refused.
    import llama_launcher.ui.panels.build_panel as bp
    from llama_launcher.core.build_spec import BuildOutput
    from llama_launcher.store.builds import add_output, load_outputs

    build_dir = tmp_path / "src" / "build-nat"
    (build_dir / "bin").mkdir(parents=True)
    binary = build_dir / "bin" / "llama-server"
    binary.write_text("x")
    add_output(
        BuildOutput(
            id="b1",
            kind="binary",
            identifier=str(binary),
            config_name="nat",
            engine="llama.cpp",
            git_ref="m",
            options={},
            created="2026-08-28",
        ),
        tmp_path,
    )
    monkeypatch.setattr(bp, "list_images_detailed", lambda *a, **k: {})

    p = _panel(qtbot, tmp_path)
    p.refresh_outputs_sync()
    monkeypatch.setattr(p, "_confirm", lambda text: True)
    p.outputs_table.setCurrentCell(0, 0)
    p.delete_selected_output()
    qtbot.waitUntil(lambda: load_outputs(tmp_path) == [], timeout=3000)
    assert not build_dir.exists()
