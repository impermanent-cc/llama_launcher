"""Hermetic autouse fixture for all UI tests.

Patches every external boundary touched by MainWindow's background timers so that
no real subprocess, network call, or filesystem read can happen during a headless test
run.  Per-test monkeypatches still win because they run after this fixture sets up.
"""
import pytest

import llama_launcher.services.runtime as _runtime
import llama_launcher.services.health as _health
import llama_launcher.services.gpu as _gpu
import llama_launcher.services.metrics as _metrics
import llama_launcher.services.registry as _registry
import llama_launcher.services.router_api as _router_api
import llama_launcher.ui.main_window as _mw
from llama_launcher.ui.controllers.launch_controller import LaunchController


@pytest.fixture(autouse=True)
def _hermetic_ui_boundaries(monkeypatch):
    # No real podman spawn from async Stop/Restart in tests. Signature must
    # accept on_error too: monitor_controller.py (not yet repointed off the
    # facade) still reaches this patch via MainWindow._spawn_async, whose
    # delegator body always forwards on_error=... explicitly (even when
    # None), not just on_done.
    monkeypatch.setattr(LaunchController, "_spawn_async",
                        lambda self, argv, on_done=None, on_error=None: None)
    monkeypatch.setattr(_runtime, "container_state", lambda name, binary, connection="": "absent")
    # The container-runtime binary (podman/docker) is a shutil.which PATH probe --
    # a real external boundary. Default it present so UI tests don't depend on the
    # host/CI image actually having podman installed (a headless CI container has
    # neither); tests that care about the missing-binary path patch it to False.
    monkeypatch.setattr(_runtime, "binary_available", lambda binary: True)
    monkeypatch.setattr(_runtime, "is_rootless", lambda binary: False)
    monkeypatch.setattr(_runtime, "stats", lambda name, binary, connection="": None)
    monkeypatch.setattr(_runtime, "started_at", lambda name, binary, connection="": None)
    monkeypatch.setattr(_runtime, "list_local_images", lambda binary, engine="llama.cpp": [])
    monkeypatch.setattr(_health, "probe_health", lambda port, timeout=1.0, **kw: "down")
    monkeypatch.setattr(_gpu, "query_gpus", lambda ssh_target="": [])
    monkeypatch.setattr(_metrics, "fetch_metrics", lambda port, timeout=1.0, **kw: {})
    monkeypatch.setattr(_metrics, "fetch_slots", lambda port, timeout=1.0, **kw: [])
    monkeypatch.setattr(_metrics, "fetch_metrics_text",
                        lambda port, timeout=1.0, **kw: "")
    monkeypatch.setattr(_metrics, "fetch_props", lambda port, timeout=1.0, **kw: None)
    # Router control plane: without these, update_status -> refresh_router_models
    # would make a real HTTP call from the test suite.
    monkeypatch.setattr(_router_api, "list_models", lambda host, port, key, **kw: [])
    monkeypatch.setattr(_router_api, "load_model", lambda *a, **kw: True)
    monkeypatch.setattr(_router_api, "unload_model", lambda *a, **kw: True)
    monkeypatch.setattr(_registry, "fetch_latest", lambda repo, prefix, timeout=10.0: None)
    yield
    # Drain any in-flight pooled monitor gather so a task can't run (and write to
    # a window) during the next test's teardown.
    from PySide6.QtCore import QThreadPool
    QThreadPool.globalInstance().waitForDone(2000)


@pytest.fixture
def main_window(qtbot, tmp_path, monkeypatch):
    """A MainWindow with an isolated base_dir() (via XDG_CONFIG_HOME), for
    tests that need a real window + panel tree (e.g. node-selection wiring)."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    w = _mw.MainWindow()
    qtbot.addWidget(w)
    return w
