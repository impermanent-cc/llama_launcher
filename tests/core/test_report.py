from llama_launcher.core.report import REPORT_SECTIONS, build_report, redact_secrets


def test_redact_secrets():
    assert "SEKRET" not in redact_secrets("podman ... --api-key SEKRET --port 8080")
    assert "SEKRET" not in redact_secrets("--api-key=SEKRET")
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
    assert "MyToken123" not in result2, (
        f"Lowercase bearer token not redacted: {result2}"
    )


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
    assert "SEKRET2" not in report, (
        "build_report did not redact api-key in logs section"
    )


def test_metrics_is_a_report_section():
    assert "metrics" in REPORT_SECTIONS


def test_build_report_includes_metrics_section():
    data = {
        "command": "podman run ...",
        "profile": "{}",
        "validation": [],
        "status_history": [],
        "runtime": "rt",
        "image": "img",
        "logs": "log",
        "metrics": "generation: 42.00 tok/s\nKV cache usage: 25%",
    }
    report = build_report(data, {s: True for s in REPORT_SECTIONS})
    assert "## Metrics" in report
    assert "42.00 tok/s" in report
    assert "KV cache usage: 25%" in report

    # togglable like every other section
    without = build_report(data, {"runtime": True})
    assert "## Metrics" not in without


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


def test_api_key_is_redacted_from_report_text():
    from llama_launcher.core.report import redact_secrets

    text = "llama-server --api-key sk-abcdef1234567890abcdef --port 8080"
    out = redact_secrets(text)
    assert "sk-abcdef1234567890abcdef" not in out
    assert "8080" in out


def test_bare_sk_token_is_redacted():
    from llama_launcher.core.report import redact_secrets

    assert "sk-supersecretvalue123456" not in redact_secrets(
        "key: sk-supersecretvalue123456"
    )


def test_sk_redaction_does_not_chew_through_ordinary_log_words():
    # The sk- pattern needs a left boundary: disk-/task-/risk- prefixed tokens
    # appear in the logs section and must survive intact.
    from llama_launcher.core.report import redact_secrets

    for text in (
        "disk-cache_enabled_true",
        "task-0000000000000001",
        "risk-assessment_score_9",
    ):
        assert redact_secrets(text) == text


def test_sk_redaction_still_catches_a_real_token_after_punctuation():
    from llama_launcher.core.report import redact_secrets

    for text in (
        "key=sk-abcdefghijklmnopqrst",
        "(sk-abcdefghijklmnopqrst)",
        "sk-abcdefghijklmnopqrst",
    ):
        assert "sk-abcdefghijklmnopqrst" not in redact_secrets(text)


def test_redact_secrets_masks_known_key_with_space():
    from llama_launcher.core.report import redact_secrets

    text = "launched with key: my secret key here in logs"
    out = redact_secrets(text, known=["my secret key"])
    assert "my secret key" not in out
    assert "***" in out
