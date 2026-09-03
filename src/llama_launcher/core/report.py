import re

REPORT_SECTIONS = ("command", "validation", "runtime", "metrics", "logs")

_REDACTIONS = (
    re.compile(r"(--api-key[= ])(\S+)"),
    re.compile(r'("api-key"\s*:\s*")([^"]*)(")'),
    re.compile(r"(?i)(authorization:\s*bearer\s+)(\S+)"),
)

# A bare router API key pasted into logs or echoed by a harness, with no
# --api-key flag or Authorization header in front of it to anchor on. The
# left boundary matters: without it this also eats "disk-cache_enabled_true"
# and "task-0000000000000001", which is exactly what logs are full of.
_SK_TOKEN = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_\-]{16,}")


def redact_secrets(text: str, known=None) -> str:
    """Mask secrets in `text`. `known` is an optional iterable of exact secret
    values to redact verbatim -- catches keys the pattern rules miss (a key
    containing a space, or a bare non-`sk-` key echoed in logs)."""
    if not text:
        return text
    text = _REDACTIONS[0].sub(r"\1***", text)
    text = _REDACTIONS[1].sub(r"\1***\3", text)
    text = _REDACTIONS[2].sub(r"\1***", text)
    text = _SK_TOKEN.sub("***", text)  # same marker as the rules above
    for secret in known or ():
        if secret and secret.strip():
            text = text.replace(secret, "***")
    return text


def build_report(data: dict, sections: dict) -> str:
    lines = ["# Llama Launcher diagnostic report", ""]
    if data.get("generated_at"):
        lines += [f"_Generated: {data['generated_at']}_", ""]

    if sections.get("command"):
        lines += [
            "## Command & profile",
            "",
            "```",
            redact_secrets(data.get("command", "")),
            "```",
            "",
            "<details><summary>Profile</summary>",
            "",
            "```json",
            redact_secrets(data.get("profile", "")),
            "```",
            "</details>",
            "",
        ]
    if sections.get("validation"):
        lines += ["## Validation & status", ""]
        for v in data.get("validation", []) or ["(none)"]:
            lines.append(f"- {v}")
        hist = " \u2192 ".join(data.get("status_history", []))
        lines += ["", f"Status history: {hist}", ""]
    if sections.get("runtime"):
        lines += [
            "## Runtime / GPU / host",
            "",
            "```",
            data.get("runtime", ""),
            "```",
            "",
        ]
    if sections.get("metrics"):
        lines += ["## Metrics", "", "```", data.get("metrics", ""), "```", ""]
    if sections.get("logs"):
        lines += [
            "## Image & recent logs",
            "",
            f"Image: `{data.get('image', '')}`",
            "",
            "```",
            redact_secrets(data.get("logs", "")),
            "```",
            "",
        ]
    return "\n".join(lines)
