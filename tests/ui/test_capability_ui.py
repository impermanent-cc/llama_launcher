import struct
from pathlib import Path
from llama_launcher.core.spec import Profile, Mount, Runtime
from llama_launcher.ui.main_window import MainWindow


def _write_moe_mtp_gguf(path):
    def kv_str(k, v):
        kb, vb = k.encode(), v.encode()
        return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb
    def kv_u32(k, v):
        kb = k.encode()
        return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 4) + struct.pack("<I", v)
    kvs = [kv_str("general.architecture", "qwen35moe"),
           kv_u32("qwen35moe.block_count", 40),
           kv_u32("qwen35moe.embedding_length", 2048),
           kv_u32("qwen35moe.expert_count", 256),
           kv_u32("qwen35moe.nextn_predict_layers", 1)]
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs)) + b"".join(kvs)
    Path(path).write_bytes(blob)


def _write_plain_gguf(path, arch="llama"):
    """A GGUF with no in-file MTP head (has_mtp_infile stays False), used to
    isolate the mtp_sibling suggestion branch from the in-file one."""
    def kv_str(k, v):
        kb, vb = k.encode(), v.encode()
        return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb
    kvs = [kv_str("general.architecture", arch)]
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs)) + b"".join(kvs)
    Path(path).write_bytes(blob)


def test_moe_mtp_model_marks_dots_suggested(qtbot, tmp_path):
    _write_moe_mtp_gguf(tmp_path / "m.gguf")
    w = MainWindow()
    qtbot.addWidget(w)
    p = Profile(name="t", image="img",
                runtime=Runtime(binary="podman", gpu_mode="cdi"),
                mounts=[Mount(host=str(tmp_path), container="/models", role="model", mode="ro")],
                model="/models/m.gguf", settings={"port": 8080})
    w._configure_panel.load_profile(p)
    # in-file MTP head -> a concrete suggestion fires on spec-type, so its
    # dot is suggested and mentions MTP in the click-apply reason.
    spec_dot = w._configure_panel._widgets["spec-type"]._dot
    assert spec_dot.text() == "●"
    assert "MTP" in spec_dot.toolTip()
    # MoE model -> n-cpu-moe is RECOMMENDED tier.
    moe_dot = w._configure_panel._widgets["n-cpu-moe"]._dot
    assert moe_dot.text() == "●"
    assert "MoE" in moe_dot.toolTip()
    # no vision sibling -> mmproj field dot is muted (N/A).
    assert w._configure_panel._mmproj_dot.text() == "○"
    assert "Not applicable" in w._configure_panel._mmproj_dot.toolTip()
    # no SWA -> muted.
    swa_dot = w._configure_panel._widgets["swa-full"]._dot
    assert swa_dot.text() == "○"
    assert "MoE" in w._configure_panel.model_meta_label.text() and "MTP" in w._configure_panel.model_meta_label.text()


def test_clicking_suggestion_dot_applies_it(qtbot, tmp_path):
    _write_moe_mtp_gguf(tmp_path / "m.gguf")
    w = MainWindow()
    qtbot.addWidget(w)
    p = Profile(name="t", image="img",
                runtime=Runtime(binary="podman", gpu_mode="cdi"),
                mounts=[Mount(host=str(tmp_path), container="/models", role="model", mode="ro")],
                model="/models/m.gguf", settings={"port": 8080, "spec-type": "none"})
    w._configure_panel.load_profile(p)
    dot = w._configure_panel._widgets["spec-type"]._dot
    assert dot.text() == "●"
    dot.click()
    assert w._configure_panel._widgets["spec-type"].value() == "draft-mtp"   # in-file MTP suggestion applied


def test_mtp_sibling_suggestion_fans_out_to_both_dots(qtbot, tmp_path):
    # No in-file MTP head, but a *mtp*.gguf sibling file -> _sug_mtp's sibling
    # branch fires ONE two-key Suggestion (spec-type + draft_model). Both the
    # spec-type catalog dot and the standalone draft-model field dot must
    # light up suggested with the SAME reason, and clicking either must
    # apply the whole suggestion.
    _write_plain_gguf(tmp_path / "m.gguf")
    (tmp_path / "m-mtp.gguf").write_bytes(b"")
    w = MainWindow()
    qtbot.addWidget(w)
    p = Profile(name="t", image="img",
                runtime=Runtime(binary="podman", gpu_mode="cdi"),
                mounts=[Mount(host=str(tmp_path), container="/models", role="model", mode="ro")],
                model="/models/m.gguf", settings={"port": 8080})
    w._configure_panel.load_profile(p)

    spec_dot = w._configure_panel._widgets["spec-type"]._dot
    draft_dot = w._configure_panel._draft_model_dot
    assert spec_dot.text() == "●"
    assert draft_dot.text() == "●"
    assert "m-mtp.gguf" in spec_dot.toolTip()
    assert spec_dot.toolTip() == draft_dot.toolTip()   # same Suggestion.text

    draft_dot.click()   # clicking the OTHER key's dot applies the WHOLE suggestion
    assert w._configure_panel._widgets["spec-type"].value() == "draft-mtp"
    assert w._configure_panel.draft_model_edit.text() == "/models/m-mtp.gguf"


def test_no_model_hides_dots(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    # default profile has no model -> every catalog widget's dot is hidden
    # (state "none": empty text, not just neutral styling).
    for k in ("spec-type", "n-cpu-moe", "swa-full"):
        assert w._configure_panel._widgets[k]._dot.text() == ""
    assert w._configure_panel._mmproj_dot.text() == ""
    assert w._configure_panel._draft_model_dot.text() == ""


def test_tier_qss_and_chip_strip_are_gone(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    assert not hasattr(w, "suggestions_strip")
    assert not hasattr(w, "family_combo")
    assert not hasattr(w, "save_preset_btn")
    from llama_launcher.ui.widgets import setting_widgets
    assert not hasattr(setting_widgets, "TIER_QSS")
    assert not hasattr(setting_widgets.SettingWidget, "set_relevance")
