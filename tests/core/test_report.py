from llama_launcher.core.report import redact_secrets, build_report, REPORT_SECTIONS


def test_redact_secrets():
    assert "SEKRET" not in redact_secrets("podman ... --api-key SEKRET --port 8080")
    assert "SEKRET" not in redact_secrets('--api-key=SEKRET')
    assert "SEKRET" not in redact_secrets('"api-key": "SEKRET"')
    assert "--port 8080" in redact_secrets("--api-key SEKRET --port 8080")


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
