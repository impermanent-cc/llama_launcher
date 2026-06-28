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


def test_model_caps_apply_tiers(qtbot, tmp_path):
    _write_moe_mtp_gguf(tmp_path / "m.gguf")
    w = MainWindow()
    qtbot.addWidget(w)
    p = Profile(name="t", image="img",
                runtime=Runtime(binary="podman", gpu_mode="cdi"),
                mounts=[Mount(host=str(tmp_path), container="/models", role="model", mode="ro")],
                model="/models/m.gguf", settings={"port": 8080})
    w.load_profile(p)
    assert w._widgets["spec-type"].property("relevance") == "recommended"   # MTP head present
    assert w._widgets["n-cpu-moe"].property("relevance") == "recommended"   # MoE
    assert w.mmproj_edit.property("relevance") == "na"                      # no vision sibling
    assert w._widgets["swa-full"].property("relevance") == "na"             # no SWA
    assert "MoE" in w.model_meta_label.text() and "MTP" in w.model_meta_label.text()


def test_suggestion_chip_applies(qtbot, tmp_path):
    from PySide6.QtWidgets import QPushButton
    _write_moe_mtp_gguf(tmp_path / "m.gguf")
    w = MainWindow()
    qtbot.addWidget(w)
    p = Profile(name="t", image="img",
                runtime=Runtime(binary="podman", gpu_mode="cdi"),
                mounts=[Mount(host=str(tmp_path), container="/models", role="model", mode="ro")],
                model="/models/m.gguf", settings={"port": 8080, "spec-type": "none"})
    w.load_profile(p)
    chips = [b for b in w.suggestions_strip.findChildren(QPushButton)]
    assert chips, "expected an MTP suggestion chip"
    chips[0].click()
    assert w._widgets["spec-type"].value() == "draft-mtp"   # in-file MTP suggestion applied


def test_no_model_is_neutral(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    # default profile has no model -> every catalog widget is NEUTRAL, no chips
    from PySide6.QtWidgets import QPushButton
    assert all(w._widgets[k].property("relevance") in (None, "neutral")
               for k in ("spec-type", "n-cpu-moe", "swa-full"))
    assert w.suggestions_strip.findChildren(QPushButton) == []
