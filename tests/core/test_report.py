from llama_launcher.core.report import redact_secrets, build_report, REPORT_SECTIONS


def test_redact_secrets():
    assert "SEKRET" not in redact_secrets("podman ... --api-key SEKRET --port 8080")
    assert "SEKRET" not in redact_secrets('--api-key=SEKRET')
    assert "SEKRET" not in redact_secrets('"api-key": "SEKRET"')
    assert "--port 8080" in redact_secrets("--api-key SEKRET --port 8080")


def test_redact_bearer_token():
    """Authorization: Bearer tokens must be masked by redact_secrets."""
    line = "Authorization: Bearer SEKRET"
    result = redact_secrets(line)
    assert "SEKRET" not in result, f"Bearer token not redacted: {result}"
    # Case-insensitive
    line2 = "authorization: bearer MyToken123"
    result2 = redact_secrets(line2)
    assert "MyToken123" not in result2, f"Lowercase bearer token not redacted: {result2}"


def test_build_report_redacts_logs():
    """build_report must redact secrets in the logs section."""
    data = {
        "command": "podman run ...",
        "profile": '{"name": "x"}',
        "validation": [],
        "status_history": [],
        "runtime": "podman 5.x rootless=true",
        "image": "ghcr.io/x:tag",
        "logs": "server started\nAuthorization: Bearer SEKRET\n--api-key SEKRET2",
    }
    report = build_report(data, {s: True for s in REPORT_SECTIONS})
    assert "SEKRET" not in report, "build_report did not redact secret in logs section"
    assert "SEKRET2" not in report, "build_report did not redact api-key in logs section"


def test_build_report_sections_toggle():
    data = {
        "generated_at": "2026-06-22T10:00:00",
        "command": "podman run ...",
        "profile": '{"name": "x"}',
        "validation": ["[warning] something"],
        "status_history": ["stopped", "running"],
        "runtime": "podman 5.x rootless=true",
        "image": "ghcr.io/...:server-cuda12-b9628",
        "logs": "loaded model",
    }
    full = build_report(data, {s: True for s in REPORT_SECTIONS})
    assert "## Command & profile" in full
    assert "## Validation & status" in full
    assert "## Runtime / GPU / host" in full
    assert "## Image & recent logs" in full

    only_runtime = build_report(data, {"runtime": True})
    assert "## Runtime / GPU / host" in only_runtime
    assert "## Command & profile" not in only_runtime
