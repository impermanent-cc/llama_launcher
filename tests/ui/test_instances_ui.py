import pytest

from llama_launcher.core.instances import Instance
from llama_launcher.core.spec import Profile, Runtime
from llama_launcher.store import profiles as store
from llama_launcher.ui.main_window import MainWindow


@pytest.fixture
def win(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _inst(name="llama-emb", profile="emb", port=8081, running=True, embeddings=True):
    return Instance(name=name, profile=profile, mode="server", running=running,
                    port=port, host="127.0.0.1", embeddings=embeddings, reranking=False)


def test_monitored_profile_falls_back_to_current_when_none(win):
    win.load_profile(Profile(name="Solo", image="img", settings={"port": 8080}))
    assert win._active_instance is None
    assert win._monitored_profile().name == "Solo"
    assert win._monitored_container_name() == win._container_name()


def test_monitored_profile_resolves_active_instance(win):
    base = store.default_base_dir()
    store.save_profile(Profile(name="emb", image="img", settings={"port": 8081}), base)
    win.load_profile(Profile(name="Solo", image="img", settings={"port": 8080}))
    win._active_instance = _inst()
    assert win._monitored_profile().name == "emb"          # not the form's "Solo"
    assert win._monitored_container_name() == "llama-emb"


def test_instance_summary_embedding_row_is_ready(win, monkeypatch):
    monkeypatch.setattr("llama_launcher.ui.main_window.health.probe_health",
                        lambda *a, **k: "ready")
    s = win.instance_summary(_inst(embeddings=True))
    assert s["health"] == "ready" and s["stat"] == "ready"


def test_instance_summary_gen_row_shows_tok_s(win, monkeypatch):
    monkeypatch.setattr("llama_launcher.ui.main_window.health.probe_health",
                        lambda *a, **k: "ready")
    monkeypatch.setattr("llama_launcher.ui.main_window.metrics.fetch_metrics",
                        lambda *a, **k: {"llamacpp:predicted_tokens_seconds": 64.0})
    s = win.instance_summary(_inst(embeddings=False))
    assert s["stat"] == "64 tok/s"
