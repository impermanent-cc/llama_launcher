from llama_launcher.core.spec import Profile, Runtime, RpcWorker
from llama_launcher.core.validation import Issue
from llama_launcher.ui.panels import configure_panel as cp


def test_rpc_mode_disables_node_and_shows_workers(main_window):
    p = main_window._configure_panel
    idx = p.launch_mode_combo.findData("rpc")
    p.launch_mode_combo.setCurrentIndex(idx)
    assert not p.node_combo.isEnabled()          # head forced local
    assert p._rpc_workers_row_visible()          # helper the panel exposes


def test_container_mode_keeps_node_enabled_and_hides_workers(main_window):
    p = main_window._configure_panel
    # Select rpc first so switching back to container (already the combo's
    # default index) still fires the mode-changed handler under test.
    p.launch_mode_combo.setCurrentIndex(p.launch_mode_combo.findData("rpc"))
    p.launch_mode_combo.setCurrentIndex(p.launch_mode_combo.findData("container"))
    assert p.node_combo.isEnabled()
    assert not p._rpc_workers_row_visible()


def test_native_mode_still_hides_workers_row(main_window):
    p = main_window._configure_panel
    idx = p.launch_mode_combo.findData("native")
    p.launch_mode_combo.setCurrentIndex(idx)
    assert not p._rpc_workers_row_visible()
    assert p.node_combo.isEnabled()


def test_build_profile_forces_local_node_and_captures_workers(main_window):
    p = main_window._configure_panel
    idx = p.launch_mode_combo.findData("rpc")
    p.launch_mode_combo.setCurrentIndex(idx)
    workers = [RpcWorker(node="local", device="CUDA0", mem_mb=8000, port=50052)]
    p.rpc_workers_table.set_workers(workers)
    profile = p.current_profile()
    assert profile.runtime.node == "local"
    assert profile.runtime.rpc_workers == workers


def test_load_profile_round_trips_rpc_workers(main_window):
    p = main_window._configure_panel
    workers = [RpcWorker(node="local", device="CPU", mem_mb=16000, port=50053)]
    profile = Profile(name="rpc-test",
                      runtime=Runtime(launch_mode="rpc", node="local",
                                      rpc_workers=workers))
    p.load_profile(profile)
    assert p.launch_mode_combo.currentData() == "rpc"
    assert not p.node_combo.isEnabled()
    assert p.rpc_workers_table.workers() == workers


def test_check_fit_button_only_shown_in_rpc_mode(main_window):
    # Select rpc first so switching to container fires the mode-changed
    # handler under test even when the combo's default index is already
    # container.
    p = main_window._configure_panel
    p.launch_mode_combo.setCurrentIndex(p.launch_mode_combo.findData("rpc"))
    assert p._check_fit_row_visible()
    p.launch_mode_combo.setCurrentIndex(p.launch_mode_combo.findData("container"))
    assert not p._check_fit_row_visible()
    p.launch_mode_combo.setCurrentIndex(p.launch_mode_combo.findData("native"))
    assert not p._check_fit_row_visible()


def test_gather_check_fit_wires_probes_into_validate(monkeypatch, tmp_path):
    """The Check-fit gather passes the probed worker_image_present /
    worker_free_mb dicts into validate(...) so the RPC warnings surface, and
    returns validate's issues alongside the headline string."""
    profile = Profile(name="pool", image="img", runtime=Runtime(
        launch_mode="rpc",
        rpc_workers=[RpcWorker(node="local", device="CUDA0", mem_mb=0)]))

    monkeypatch.setattr(cp.pool_preflight, "default_gpus_reader",
                        lambda base: (lambda n: 10 * 1024 ** 3))
    monkeypatch.setattr(cp.pool_preflight, "default_ram_reader",
                        lambda base: (lambda n: 0))
    monkeypatch.setattr(cp.pool_preflight, "gather_donations",
                        lambda profile, base, gpus, ram: [("vram", 10 * 1024 ** 3)])
    monkeypatch.setattr(cp.pool_preflight, "headline", lambda est, dons: "HEADLINE")
    monkeypatch.setattr(cp.runtime, "image_exists",
                        lambda image, binary, connection="": False)
    monkeypatch.setattr(cp.runtime, "binary_available", lambda binary: True)
    monkeypatch.setattr(cp.native, "native_binary_ok_for", lambda p: True)

    captured = {}

    def fake_validate(profile, **kw):
        captured.update(kw)
        return [Issue("error", "boom")]
    monkeypatch.setattr(cp, "validate", fake_validate)

    text, issues = cp._gather_check_fit(profile, tmp_path, 5 * 1024 ** 3)

    assert text == "HEADLINE"
    assert [i.message for i in issues] == ["boom"]
    assert captured["worker_image_present"] == {"local": False}
    assert captured["worker_free_mb"] == {"local": 10 * 1024}


def test_check_fit_click_renders_headline_and_issues(main_window, monkeypatch, qtbot):
    p = main_window._configure_panel
    p.launch_mode_combo.setCurrentIndex(p.launch_mode_combo.findData("rpc"))
    p.rpc_workers_table.set_workers(
        [RpcWorker(node="local", device="CUDA0", mem_mb=0, port=50052)])
    p.model_edit.setText("/models/x.gguf")
    monkeypatch.setattr(cp, "_gather_check_fit",
                        lambda profile, base_dir, estimate: (
                            "HEADLINE TEXT", [Issue("warning", "watch out")]))

    p._on_check_fit()
    qtbot.waitUntil(lambda: not p._check_fit_inflight, timeout=3000)
    qtbot.waitUntil(lambda: "HEADLINE TEXT" in p.check_fit_label.text(), timeout=3000)
    assert "watch out" in p.check_fit_label.text()


def test_fresh_window_hides_launch_mode_rows_in_default_container_mode(main_window):
    # Startup loads no profile and the launch-mode combo defaults to
    # "container", so currentIndexChanged never fires; the native-only and
    # RPC-only rows must still start hidden.
    p = main_window._configure_panel
    assert p.launch_mode_combo.currentData() == "container"
    assert not p._rpc_workers_row_visible()
    assert not p._check_fit_row_visible()
    index = p._left_form.getWidgetPosition(p._native_binary_row)[0]
    assert index >= 0 and not p._left_form.isRowVisible(index)
    assert p.node_combo.isEnabled()
