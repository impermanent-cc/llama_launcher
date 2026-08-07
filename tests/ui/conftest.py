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


@pytest.fixture(autouse=True)
def _hermetic_ui_boundaries(monkeypatch):
    # No real podman spawn from async Stop/Restart in tests.
    monkeypatch.setattr(_mw.MainWindow, "_spawn_async",
                        lambda self, argv, on_done=None: None)
    monkeypatch.setattr(_runtime, "container_state", lambda name, binary: "absent")
    monkeypatch.setattr(_runtime, "is_rootless", lambda binary: False)
    monkeypatch.setattr(_runtime, "stats", lambda name, binary: None)
    monkeypatch.setattr(_runtime, "started_at", lambda name, binary: None)
    monkeypatch.setattr(_runtime, "list_local_images", lambda binary: [])
    monkeypatch.setattr(_health, "probe_health", lambda port, timeout=1.0, **kw: "down")
    monkeypatch.setattr(_gpu, "query_gpus", lambda: [])
    monkeypatch.setattr(_gpu, "free_vram_bytes", lambda: None)
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
