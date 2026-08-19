from pathlib import Path

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


def test_is_alive_true_for_self(tmp_path):
    import os, sys
    # This test process's cmdline references the python executable.
    assert native.is_alive(os.getpid(), sys.executable) is True


def test_list_native_instances_shape_and_prune(tmp_path):
    alive = {"pid": _self_pid(), "profile": "Gen", "port": 8080, "host": "127.0.0.1",
             "started_at": "t", "binary": _self_exe(),
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
    assert r["kind"] == "native" and r["pid"] == _self_pid()
    # dead entry's registry file was pruned
    assert not (native.registry_dir(tmp_path) / "dead.json").exists()


def _self_pid():
    import os
    return os.getpid()


def _self_exe():
    import sys
    return sys.executable
