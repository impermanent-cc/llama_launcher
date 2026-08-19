import signal
import subprocess
from pathlib import Path

from llama_launcher.core.spec import Profile, Runtime
from llama_launcher.services import native


def test_write_and_read_entry_round_trips(tmp_path):
    entry = {"pid": 111, "profile": "Gen", "port": 8080, "host": "127.0.0.1",
             "started_at": "2026-08-19T00:00:00", "binary": "/opt/llama-server",
             "log": str(tmp_path / "native" / "gen.log")}
    native.write_entry(tmp_path, entry)
    got = native.read_entries(tmp_path)
    assert got == [entry]


def test_read_entries_skips_corrupt_file(tmp_path):
    d = native.registry_dir(tmp_path)
    d.mkdir(parents=True)
    (d / "bad.json").write_text("{not json")
    assert native.read_entries(tmp_path) == []


def test_is_alive_false_for_missing_pid(tmp_path):
    # PID 1 exists but its cmdline won't reference our fake binary.
    assert native.is_alive(1, "/opt/nonexistent-llama-server") is False


def test_is_alive_true_for_self():
    import sys
    # Spawn a child launched by ABSOLUTE path so its cmdline argv[0] == sys.executable,
    # matching how launch_native spawns a server by absolute native_binary path.
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert native.is_alive(proc.pid, sys.executable) is True
        assert native.is_alive(proc.pid, "/opt/nonexistent-binary") is False
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait()


def test_list_native_instances_shape_and_prune(tmp_path):
    import sys
    # Spawn a child launched by ABSOLUTE path so its cmdline argv[0] matches the binary.
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        alive = {"pid": proc.pid, "profile": "Gen", "port": 8080, "host": "127.0.0.1",
                 "started_at": "t", "binary": sys.executable,
                 "log": str(tmp_path / "native" / "gen.log")}
        dead = {"pid": 999999, "profile": "Dead", "port": 8081, "host": "127.0.0.1",
                "started_at": "t", "binary": "/opt/gone", "log": "x"}
        native.write_entry(tmp_path, alive)
        native.write_entry(tmp_path, dead)
        rows = native.list_native_instances(tmp_path)
        assert len(rows) == 1
        r = rows[0]
        assert r["name"] == "llama-gen" and r["profile"] == "Gen"
        assert r["running"] is True and r["mode"] == "server"
        assert r["kind"] == "native" and r["pid"] == proc.pid
        # dead entry's registry file was pruned
        assert not (native.registry_dir(tmp_path) / "dead.json").exists()
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait()


def test_launch_native_spawns_and_registers(tmp_path, monkeypatch):
    from llama_launcher.core.spec import Profile, Runtime
    # Point build_command at a real, harmless long-lived process instead of a
    # llama-server: patch it to return a `sleep` argv so the smoke is hermetic.
    monkeypatch.setattr(native, "build_command",
                        lambda p, **k: ["sleep", "30"])
    p = Profile(name="Gen", runtime=Runtime(launch_mode="native",
                native_binary="sleep", bind_host="127.0.0.1"),
                settings={"port": 8080})
    res = native.launch_native(p, tmp_path, now_iso="2026-08-19T00:00:00")
    try:
        assert res.ok and res.pid and res.name == "llama-gen"
        entries = native.read_entries(tmp_path)
        assert entries and entries[0]["pid"] == res.pid
        assert native.native_log_path(tmp_path, "Gen").exists()
    finally:
        if res.pid:
            native.stop_native(res.pid, signal.SIGKILL)


def test_stop_native_swallows_missing_pid():
    native.stop_native(999999, signal.SIGTERM)  # must not raise


def test_remove_native_deletes_registry_and_log(tmp_path):
    entry = {"pid": 1, "profile": "Gen", "port": 8080, "host": "127.0.0.1",
             "started_at": "t", "binary": "x",
             "log": str(native.native_log_path(tmp_path, "Gen"))}
    native.write_entry(tmp_path, entry)
    native.native_log_path(tmp_path, "Gen").parent.mkdir(parents=True, exist_ok=True)
    native.native_log_path(tmp_path, "Gen").write_text("log")
    native.remove_native("llama-gen", tmp_path)
    assert native.read_entries(tmp_path) == []
    assert not native.native_log_path(tmp_path, "Gen").exists()


def test_logs_argv():
    assert native.logs_argv("/x/gen.log") == ["tail", "-n", "200", "-f", "/x/gen.log"]


def test_proc_stats_none_for_missing_pid():
    assert native.proc_stats(999999, interval=0.0) is None


def test_proc_stats_reports_for_self():
    import os
    st = native.proc_stats(os.getpid(), interval=0.0)
    assert st is not None
    assert st["mem_usage"].endswith(" MiB")
    assert st["cpu_perc"].endswith("%")


def test_native_binary_available_false_for_missing_path():
    assert native.native_binary_available("/no/such/llama-server") is False


def test_native_binary_available_false_for_empty_path():
    assert native.native_binary_available("") is False


def test_native_binary_available_false_for_non_executable_file(tmp_path):
    f = tmp_path / "llama-server"
    f.write_text("#!/bin/sh\n")
    f.chmod(0o644)
    assert native.native_binary_available(str(f)) is False


def test_native_binary_available_true_for_executable_file(tmp_path):
    f = tmp_path / "llama-server"
    f.write_text("#!/bin/sh\n")
    f.chmod(0o755)
    assert native.native_binary_available(str(f)) is True


def test_native_binary_ok_for_ignores_container_profiles():
    p = Profile(name="c", runtime=Runtime(launch_mode="container", native_binary=""))
    assert native.native_binary_ok_for(p) is True


def test_native_binary_ok_for_stats_native_profile(tmp_path):
    missing = Profile(name="n", runtime=Runtime(launch_mode="native",
                       native_binary="/no/such/llama-server"))
    assert native.native_binary_ok_for(missing) is False

    exe = tmp_path / "llama-server"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    present = Profile(name="n", runtime=Runtime(launch_mode="native",
                       native_binary=str(exe)))
    assert native.native_binary_ok_for(present) is True
