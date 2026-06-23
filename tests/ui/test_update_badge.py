import llama_launcher.ui.main_window as mw
from llama_launcher.core.spec import Profile, Runtime


def test_check_for_update_finds_newer(qtbot):
    w = mw.MainWindow(); qtbot.addWidget(w)
    w.load_profile(Profile(name="u", image="ghcr.io/ggml-org/llama.cpp:server-cuda12-b9628",
                           runtime=Runtime(binary="podman"), settings={"port": 8080}))
    newer = w.check_for_update(["server-cuda12-b9628", "server-cuda12-b9755", "buildcache-x"])
    assert newer == "server-cuda12-b9755"


def test_check_for_update_none_when_current(qtbot):
    w = mw.MainWindow(); qtbot.addWidget(w)
    w.load_profile(Profile(name="u", image="ghcr.io/ggml-org/llama.cpp:server-cuda12-b9755",
                           runtime=Runtime(binary="podman"), settings={"port": 8080}))
    assert w.check_for_update(["server-cuda12-b9628", "server-cuda12-b9755"]) is None
