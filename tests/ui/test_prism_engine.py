from llama_launcher.core.capabilities import derive_caps
from llama_launcher.core.gguf import GgufMeta, TensorInfo
from llama_launcher.ui.panels import configure_panel as cp


def _row_visible(panel, key) -> bool:
    form, widget = panel._setting_rows[key]
    return form.isRowVisible(form.getWidgetPosition(widget)[0])


def test_engine_combo_offers_prism_with_its_label(main_window):
    combo = main_window._configure_panel.engine_combo
    idx = combo.findData("prism_llama.cpp")
    assert idx >= 0 and combo.itemText(idx) == "PrismML llama.cpp"
    assert [combo.itemData(i) for i in range(combo.count())] == [
        "llama.cpp",
        "ik_llama.cpp",
        "prism_llama.cpp",
    ]


def test_prism_form_hides_post_10685_flags_and_shows_its_own(main_window):
    panel = main_window._configure_panel
    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("server"))
    panel.engine_combo.setCurrentIndex(panel.engine_combo.findData("prism_llama.cpp"))
    assert not _row_visible(panel, "n-cpu-ffn")
    assert not _row_visible(panel, "tensor-read-lazy")
    assert _row_visible(panel, "kv-mean-center")
    assert _row_visible(panel, "mlock")
    assert _row_visible(panel, "load-mode")
    panel.engine_combo.setCurrentIndex(panel.engine_combo.findData("llama.cpp"))
    assert _row_visible(panel, "n-cpu-ffn")
    assert not _row_visible(panel, "kv-mean-center")


def test_prism_has_no_default_image_and_says_so(main_window):
    panel = main_window._configure_panel
    panel.engine_combo.setCurrentIndex(panel.engine_combo.findData("llama.cpp"))
    panel.image_edit.setText("")
    panel._maybe_seed_default_image("llama.cpp")
    assert panel.image_edit.text() == cp._ENGINE_DEFAULT_IMAGE["llama.cpp"]
    panel.engine_combo.setCurrentIndex(panel.engine_combo.findData("prism_llama.cpp"))
    assert panel.image_edit.text() == ""
    assert panel.image_edit.placeholderText() == cp.PRISM_IMAGE_HINT
    assert "Build tab" in cp.PRISM_IMAGE_HINT
    panel.engine_combo.setCurrentIndex(panel.engine_combo.findData("llama.cpp"))
    assert panel.image_edit.text() == cp._ENGINE_DEFAULT_IMAGE["llama.cpp"]
    assert panel.image_edit.placeholderText() == ""


def test_prism_keeps_a_user_typed_image(main_window):
    panel = main_window._configure_panel
    panel.engine_combo.setCurrentIndex(panel.engine_combo.findData("llama.cpp"))
    panel.image_edit.setText("localhost/mine:1")
    panel.engine_combo.setCurrentIndex(panel.engine_combo.findData("prism_llama.cpp"))
    assert panel.image_edit.text() == "localhost/mine:1"


def _prism_meta():
    return GgufMeta(
        arch="qwen35",
        n_layers=1,
        tensors=(TensorInfo("blk.0.attn_q.weight", 128, 143, 28),),
    )


def test_prism_weights_light_the_engine_dot_and_a_click_switches(
    main_window, monkeypatch
):
    panel = main_window._configure_panel
    meta = _prism_meta()
    monkeypatch.setattr(
        cp.model_info,
        "inspect_model",
        lambda path, mounts: (meta, 28, derive_caps(meta, [])),
    )
    panel.engine_combo.setCurrentIndex(panel.engine_combo.findData("llama.cpp"))
    panel.model_edit.setText("/models/bonsai.gguf")
    panel.apply_model_caps()
    assert panel._engine_dot.text() == "\u25cf"
    assert "PrismML" in panel._engine_dot.toolTip()
    panel._engine_dot.click()
    assert panel.engine_combo.currentData() == "prism_llama.cpp"
    assert panel._engine_dot.text() != "\u25cf"


def test_prism_weights_under_mainline_raise_the_preflight_warning(
    main_window, monkeypatch
):
    panel = main_window._configure_panel
    meta = _prism_meta()
    monkeypatch.setattr(
        cp.model_info,
        "inspect_model",
        lambda path, mounts: (meta, 28, derive_caps(meta, [])),
    )
    panel.engine_combo.setCurrentIndex(panel.engine_combo.findData("llama.cpp"))
    panel.model_edit.setText("/models/bonsai.gguf")
    panel.apply_model_caps()
    messages = [i.message for i in panel.router_issues()]
    assert any(m.startswith("PTQ1_0 weights:") for m in messages)
    panel.engine_combo.setCurrentIndex(panel.engine_combo.findData("prism_llama.cpp"))
    messages = [i.message for i in panel.router_issues()]
    assert not any(m.startswith("PTQ1_0 weights:") for m in messages)


def test_loading_profiles_keeps_the_image_hint_in_step_with_the_engine(main_window):
    from llama_launcher.core.spec import Profile, Runtime

    panel = main_window._configure_panel
    prism = Profile(
        name="bonsai",
        image="",
        runtime=Runtime(engine="prism_llama.cpp"),
        model="/models/bonsai.gguf",
    )
    panel.load_profile(prism)
    assert panel.engine_combo.currentData() == "prism_llama.cpp"
    assert panel.image_edit.text() == ""
    assert panel.image_edit.placeholderText() == cp.PRISM_IMAGE_HINT
    mainline = Profile(name="ml", image="", model="/models/m.gguf")
    panel.load_profile(mainline)
    assert panel.engine_combo.currentData() == "llama.cpp"
    assert panel.image_edit.placeholderText() == ""
    assert panel.image_edit.text() == ""


def test_image_hint_is_for_prism_alone(main_window):
    panel = main_window._configure_panel
    panel._apply_image_placeholder("prism_llama.cpp")
    assert panel.image_edit.placeholderText() == cp.PRISM_IMAGE_HINT
    panel._apply_image_placeholder("future_llama.cpp")
    assert panel.image_edit.placeholderText() == ""
    panel._apply_image_placeholder("llama.cpp")
    assert panel.image_edit.placeholderText() == ""
