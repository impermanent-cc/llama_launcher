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
