"""Pre-launch warnings offer a way OUT: every warning dialog (VRAM check,
validation warnings, router preset warnings) asks Abort/Ignore, with Abort as
the default, so a launch that is predicted to OOM can be cancelled instead of
watched fail.
"""
from PySide6.QtWidgets import QMessageBox

import llama_launcher.ui.controllers.launch_controller as lc
from llama_launcher.core.spec import Profile, Mount, Runtime
from llama_launcher.core.validation import Issue


def _server_profile():
    return Profile(name="v", image="img", runtime=Runtime(binary="podman"),
                   mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
                   model="/models/m.gguf", settings={"port": 8080})


def _router_profile():
    return Profile(name="r", image="img", runtime=Runtime(binary="podman"),
                   mode="router", settings={"port": 8080})


def _answer_warning(monkeypatch, answer, captured=None):
    def _warning(parent, title, text, *rest):
        if captured is not None:
            captured.append((title, text, rest))
        return answer
    monkeypatch.setattr(lc.QMessageBox, "warning", staticmethod(_warning))


def _track_terminal(monkeypatch):
    launched = []
    monkeypatch.setattr(lc.terminal, "launch",
                        lambda argv, template=None: launched.append(argv))
    return launched


# -- VRAM check ---------------------------------------------------------------

def test_vram_warning_abort_stops_launch(main_window, monkeypatch):
    ctl = main_window._launch
    main_window._configure_panel.load_profile(_server_profile())
    monkeypatch.setattr(ctl, "vram_check", lambda: "won't fit")
    _answer_warning(monkeypatch, QMessageBox.Abort)
    launched = _track_terminal(monkeypatch)

    ctl.on_launch()

    assert launched == []


def test_vram_warning_ignore_proceeds(main_window, monkeypatch):
    ctl = main_window._launch
    main_window._configure_panel.load_profile(_server_profile())
    monkeypatch.setattr(ctl, "vram_check", lambda: "won't fit")
    _answer_warning(monkeypatch, QMessageBox.Ignore)
    launched = _track_terminal(monkeypatch)

    ctl.on_launch()

    assert len(launched) == 1


def test_vram_warning_defaults_to_abort(main_window, monkeypatch):
    """Enter must back out, not launch: Abort is the default button."""
    ctl = main_window._launch
    main_window._configure_panel.load_profile(_server_profile())
    monkeypatch.setattr(ctl, "vram_check", lambda: "won't fit")
    captured = []
    _answer_warning(monkeypatch, QMessageBox.Abort, captured)
    _track_terminal(monkeypatch)

    ctl.on_launch()

    (title, text, rest) = captured[0]
    assert title == "VRAM check"
    assert "won't fit" in text
    assert QMessageBox.Abort in rest          # default button argument


# -- validation warnings ------------------------------------------------------

def test_validation_warning_abort_stops_launch(main_window, monkeypatch):
    ctl = main_window._launch
    main_window._configure_panel.load_profile(_server_profile())
    monkeypatch.setattr(main_window._configure_panel, "router_issues",
                        lambda: [Issue("warning", "port already in use")])
    _answer_warning(monkeypatch, QMessageBox.Abort)
    launched = _track_terminal(monkeypatch)

    ctl.on_launch()

    assert launched == []


def test_validation_warning_ignore_proceeds(main_window, monkeypatch):
    ctl = main_window._launch
    main_window._configure_panel.load_profile(_server_profile())
    monkeypatch.setattr(main_window._configure_panel, "router_issues",
                        lambda: [Issue("warning", "port already in use")])
    monkeypatch.setattr(ctl, "vram_check", lambda: None)
    _answer_warning(monkeypatch, QMessageBox.Ignore)
    launched = _track_terminal(monkeypatch)

    ctl.on_launch()

    assert len(launched) == 1


# -- router preset warnings ---------------------------------------------------

def test_preset_warning_abort_stops_router_launch(main_window, monkeypatch, tmp_path):
    ctl = main_window._launch
    main_window._configure_panel.load_profile(_router_profile())
    # Validation is not under test here; an unstubbed run pops a real
    # (blocking) critical box for the member-less router profile.
    monkeypatch.setattr(ctl, "_validate_or_warn", lambda: True)
    monkeypatch.setattr(main_window, "prepare_router_files",
                        lambda: (str(tmp_path), ["preset x is stale"]))
    built = []
    monkeypatch.setattr(lc, "build_command",
                        lambda p, **kw: built.append(p) or ["podman"])
    _answer_warning(monkeypatch, QMessageBox.Abort)

    ctl.on_launch()

    assert built == []


def test_preset_warning_ignore_proceeds_with_router_launch(main_window, monkeypatch, tmp_path):
    ctl = main_window._launch
    main_window._configure_panel.load_profile(_router_profile())
    monkeypatch.setattr(ctl, "_validate_or_warn", lambda: True)
    monkeypatch.setattr(main_window, "prepare_router_files",
                        lambda: (str(tmp_path), ["preset x is stale"]))
    built = []
    monkeypatch.setattr(lc, "build_command",
                        lambda p, **kw: built.append(p) or ["podman"])
    _answer_warning(monkeypatch, QMessageBox.Ignore)

    ctl.on_launch()

    assert len(built) == 1
