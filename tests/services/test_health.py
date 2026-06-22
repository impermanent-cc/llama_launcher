from llama_launcher.services.health import derive_status


def test_absent_is_stopped():
    assert derive_status("absent", False) == "stopped"


def test_running_container_not_ready_is_starting():
    assert derive_status("running", False) == "starting"


def test_running_and_healthy_is_running():
    assert derive_status("running", True) == "running"


def test_stopped_container_is_error_when_expected_up():
    # container exists but is stopped while we expected it running
    assert derive_status("stopped", False) == "error"
