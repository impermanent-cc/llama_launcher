"""RPC pool launch fit gate: launching a pool with a model runs the pooled
VRAM+RAM preflight (the manual "Check fit" probe) off-thread first; a
doesn't-fit verdict asks Abort/Ignore before the pool spins up. Fail-open:
a broken probe must never block a launch.
"""
from PySide6.QtWidgets import QMessageBox

import llama_launcher.services.rpc as rpc
import llama_launcher.ui.controllers.launch_controller as lc
from llama_launcher.core.spec import Profile, Mount, Runtime

GIB = 1024 ** 3


def _rpc_model_profile():
    return Profile(name="pool", image="img:tag",
                   runtime=Runtime(launch_mode="rpc"),
                   mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
                   model="/models/m.gguf", settings={"port": 8080})


def _wire(main_window, monkeypatch, *, donations, estimate=10 * GIB, answer=None):
    """Common harness: sync pool seam, patched probes, recorded launch_pool."""
    ctl = main_window._launch
    main_window._configure_panel.load_profile(_rpc_model_profile())
    monkeypatch.setattr(ctl, "_validate_or_warn", lambda: True)
    monkeypatch.setattr(ctl, "_run_pool_async",
                        lambda work, on_done: on_done(work()))
    monkeypatch.setattr(main_window._configure_panel, "_model_estimate_bytes",
                        lambda p: estimate)
    if isinstance(donations, Exception):
        def _gather(p, base, **kw):
            raise donations
    else:
        def _gather(p, base, **kw):
            return donations
    monkeypatch.setattr(lc.pool_preflight, "gather_donations", _gather)
    launched = []
    monkeypatch.setattr(rpc, "launch_pool",
                        lambda p, base, **kw: launched.append(p.name) or rpc.PoolResult(True))
    if answer is not None:
        monkeypatch.setattr(lc.QMessageBox, "warning",
                            staticmethod(lambda *a, **kw: answer))
    return ctl, launched


def test_pool_launch_proceeds_when_fits(main_window, monkeypatch):
    ctl, launched = _wire(main_window, monkeypatch,
                          donations=[("vram", 20 * GIB)])
    ctl.on_launch()
    assert launched == ["pool"]


def test_pool_launch_short_abort_stops(main_window, monkeypatch):
    ctl, launched = _wire(main_window, monkeypatch,
                          donations=[("vram", 5 * GIB)],
                          answer=QMessageBox.Abort)
    ctl.on_launch()
    assert launched == []


def test_pool_launch_short_ignore_proceeds(main_window, monkeypatch):
    ctl, launched = _wire(main_window, monkeypatch,
                          donations=[("vram", 5 * GIB)],
                          answer=QMessageBox.Ignore)
    ctl.on_launch()
    assert launched == ["pool"]


def test_pool_launch_probe_failure_fails_open(main_window, monkeypatch):
    ctl, launched = _wire(main_window, monkeypatch,
                          donations=OSError("node unreachable"))
    ctl.on_launch()
    assert launched == ["pool"]


def test_pool_fit_dialog_shows_headline(main_window, monkeypatch):
    captured = []
    ctl, launched = _wire(main_window, monkeypatch,
                          donations=[("vram", 5 * GIB), ("ram", 2 * GIB)])
    monkeypatch.setattr(
        lc.QMessageBox, "warning",
        staticmethod(lambda parent, title, text, *rest:
                     captured.append((title, text)) or QMessageBox.Abort))
    ctl.on_launch()
    title, text = captured[0]
    assert title == "Pool fit"
    assert "does not fit" in text
