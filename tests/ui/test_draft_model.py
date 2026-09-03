"""draft_model picker wiring in the GUI."""

from llama_launcher.core.spec import Mount, Profile, Runtime
from llama_launcher.ui.main_window import MainWindow


def _base_profile(**kwargs):
    return Profile(
        name="draft-test",
        image="img:tag",
        runtime=Runtime(binary="podman", gpu_mode="cdi"),
        mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
        model="/models/m.gguf",
        settings={"port": 8080},
        **kwargs,
    )


def test_draft_model_roundtrip(qtbot):
    """Loading a profile with draft_model must survive the round-trip through current_profile()."""
    w = MainWindow()
    qtbot.addWidget(w)
    p = _base_profile(draft_model="/models/d.gguf")
    w._configure_panel.load_profile(p)
    out = w._configure_panel.current_profile()
    assert out.draft_model == "/models/d.gguf", (
        "draft_model data was lost (not wired into current_profile)"
    )


def test_draft_model_appears_in_preview(qtbot):
    """--spec-draft-model must appear in the command preview when draft_model is set."""
    w = MainWindow()
    qtbot.addWidget(w)
    p = _base_profile(draft_model="/models/d.gguf")
    w._configure_panel.load_profile(p)
    text = w._configure_panel.preview_text()
    assert "--spec-draft-model /models/d.gguf" in text, (
        f"Expected '--spec-draft-model /models/d.gguf' in preview, got:\n{text}"
    )


def test_draft_model_edit_widget_exists(qtbot):
    """MainWindow must expose a draft_model_edit QLineEdit attribute."""
    w = MainWindow()
    qtbot.addWidget(w)
    from PySide6.QtWidgets import QLineEdit

    assert hasattr(w._configure_panel, "draft_model_edit"), (
        "draft_model_edit attribute missing"
    )
    assert isinstance(w._configure_panel.draft_model_edit, QLineEdit)


def test_draft_model_clears_on_empty_profile(qtbot):
    """Loading a profile with no draft_model must clear the draft_model field."""
    w = MainWindow()
    qtbot.addWidget(w)
    # First load one with a draft model
    w._configure_panel.load_profile(_base_profile(draft_model="/models/d.gguf"))
    # Then load one without
    w._configure_panel.load_profile(_base_profile(draft_model=None))
    out = w._configure_panel.current_profile()
    assert out.draft_model is None, (
        "draft_model should be None after loading a profile without one"
    )
