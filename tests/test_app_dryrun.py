"""
TDD tests for the --dry-run CLI mode in llama_launcher.app.

These tests are written BEFORE the implementation (RED phase).
All tests should fail until dry_run() and the refactored main() are implemented.
"""
import shlex

import pytest

from llama_launcher.core.spec import Mount, Profile, Runtime
from llama_launcher.store.profiles import save_config, save_profile

# Import the functions under test; these don't exist yet, so import errors are expected.
from llama_launcher.app import dry_run, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(name="test-model", port=8080) -> Profile:
    """A profile whose model is under a mount (passes validation)."""
    return Profile(
        name=name,
        image="ghcr.io/ggml-org/llama.cpp:server",
        runtime=Runtime(binary="podman", gpu_mode="none"),
        mounts=[Mount(host="/data/models", container="/models", role="model", mode="ro")],
        model="/models/tinyllama.gguf",
        settings={"port": port},
    )


# ---------------------------------------------------------------------------
# Test 1: dry_run with a valid saved profile returns 0 and prints podman cmd
# ---------------------------------------------------------------------------

def test_dry_run_valid_profile_returns_0_and_prints_commands(
    tmp_path, capsys, monkeypatch
):
    """dry_run with a valid profile (model under a mount, port=8080) returns 0
    and stdout contains the podman container command (--host 0.0.0.0) and
    the konsole terminal line."""
    p = _make_profile("test-model", port=8080)
    save_profile(p, tmp_path)

    monkeypatch.setattr("llama_launcher.app.binary_available", lambda _: True)

    ret = dry_run(profile_name="test-model", base_dir=tmp_path)

    assert ret == 0
    out = capsys.readouterr().out

    # Container command section present
    assert "# Container command:" in out
    # Terminal invocation section present
    assert "# Terminal invocation:" in out

    # podman command includes essential parts
    assert "podman" in out
    assert "--host 0.0.0.0" in out
    assert "--port 8080" in out

    # Terminal invocation wraps in konsole
    assert "konsole" in out


# ---------------------------------------------------------------------------
# Test 2: dry_run with an unknown --profile name returns 2 and lists names
# ---------------------------------------------------------------------------

def test_dry_run_unknown_profile_returns_2_and_lists_available(
    tmp_path, capsys, monkeypatch
):
    """dry_run with a profile name that doesn't exist returns 2 and prints
    available profile names."""
    save_profile(_make_profile("alpha"), tmp_path)
    save_profile(_make_profile("beta"), tmp_path)

    monkeypatch.setattr("llama_launcher.app.binary_available", lambda _: True)

    ret = dry_run(profile_name="does-not-exist", base_dir=tmp_path)

    assert ret == 2
    out = capsys.readouterr().out
    assert "does-not-exist" in out
    # Both existing names must be listed
    assert "alpha" in out
    assert "beta" in out


# ---------------------------------------------------------------------------
# Test 3: dry_run with no profiles in base_dir returns 2
# ---------------------------------------------------------------------------

def test_dry_run_no_profiles_returns_2(tmp_path, capsys):
    """dry_run with an empty profiles directory returns 2 and prints a
    helpful message mentioning the base_dir."""
    ret = dry_run(profile_name=None, base_dir=tmp_path)

    assert ret == 2
    out = capsys.readouterr().out
    assert "No saved profiles" in out
    assert str(tmp_path) in out


# ---------------------------------------------------------------------------
# Test 4: dry_run on a profile with no model (validation error) returns 1
# ---------------------------------------------------------------------------

def test_dry_run_validation_error_returns_1_and_prints_error(
    tmp_path, capsys, monkeypatch
):
    """dry_run on a profile with no model set triggers a validation error,
    returns 1, and prints the validation issues under a '# Validation:' header."""
    # Profile with no model; validation will produce an error
    p = Profile(
        name="no-model",
        image="ghcr.io/ggml-org/llama.cpp:server",
        runtime=Runtime(binary="podman", gpu_mode="none"),
    )
    save_profile(p, tmp_path)

    monkeypatch.setattr("llama_launcher.app.binary_available", lambda _: True)

    ret = dry_run(profile_name="no-model", base_dir=tmp_path)

    assert ret == 1
    out = capsys.readouterr().out
    assert "# Validation:" in out
    assert "[error]" in out


# ---------------------------------------------------------------------------
# Test 5: main(["--dry-run", "--profile", NAME]) routes to dry_run
# ---------------------------------------------------------------------------

def test_main_dry_run_flag_routes_to_dry_run(tmp_path, capsys, monkeypatch):
    """main() with --dry-run and --profile routes to dry_run() and returns
    its exit code without touching Qt."""
    p = _make_profile("cli-profile")
    save_profile(p, tmp_path)

    monkeypatch.setattr("llama_launcher.app.binary_available", lambda _: True)
    # Make default_base_dir() return tmp_path so main() finds our profiles.
    monkeypatch.setattr("llama_launcher.app.default_base_dir", lambda: tmp_path)

    ret = main(["--dry-run", "--profile", "cli-profile"])

    assert ret == 0
    out = capsys.readouterr().out
    assert "# Container command:" in out
    assert "# Terminal invocation:" in out


def test_dry_run_redacts_api_key(tmp_path, capsys, monkeypatch):
    """--dry-run output is what users paste into bug reports; the api-key must
    not appear in it (neither the container command nor the terminal line)."""
    p = _make_profile("keyed", port=8080)
    p.settings["api-key"] = "sk-supersecret-value"
    save_profile(p, tmp_path)
    monkeypatch.setattr("llama_launcher.app.binary_available", lambda _: True)
    ret = dry_run(profile_name="keyed", base_dir=tmp_path)
    out = capsys.readouterr().out
    assert ret in (0, 1)
    assert "sk-supersecret-value" not in out
    assert "***" in out
