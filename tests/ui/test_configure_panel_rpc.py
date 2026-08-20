from llama_launcher.core.spec import Profile, Runtime, RpcWorker


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
