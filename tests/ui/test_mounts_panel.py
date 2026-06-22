from llama_launcher.core.spec import Mount
from llama_launcher.ui.panels.mounts_panel import MountsPanel


def test_mounts_panel_roundtrip(qtbot):
    panel = MountsPanel()
    qtbot.addWidget(panel)
    mounts = [Mount(host="/h/models", container="/models", role="model", mode="ro"),
              Mount(host="/h/ws", container="/workspace", role="workspace",
                    mode="rw", workdir=True)]
    panel.set_mounts(mounts)
    out = panel.mounts()
    assert out[0].container == "/models" and out[0].mode == "ro"
    assert out[1].workdir is True and out[1].mode == "rw"
