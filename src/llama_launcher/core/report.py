import re

REPORT_SECTIONS = ("command", "validation", "runtime", "logs")

_REDACTIONS = (
    re.compile(r"(--api-key[= ])(\S+)"),
    re.compile(r'("api-key"\s*:\s*")([^"]*)(")'),
    re.compile(r"(?i)(authorization:\s*bearer\s+)(\S+)"),
)


def redact_secrets(text: str) -> str:
    if not text:
        return text
    text = _REDACTIONS[0].sub(r"\1***", text)
    text = _REDACTIONS[1].sub(r"\1***\3", text)
    text = _REDACTIONS[2].sub(r"\1***", text)
    return text


def build_report(data: dict, sections: dict) -> str:
    lines = ["# Llama Launcher diagnostic report", ""]
    if data.get("generated_at"):
        lines += [f"_Generated: {data['generated_at']}_", ""]

    if sections.get("command"):
        lines += ["## Command & profile", "",
                  "```", redact_secrets(data.get("command", "")), "```", "",
                  "<details><summary>Profile</summary>", "",
                  "```json", redact_secrets(data.get("profile", "")), "```",
                  "</details>", ""]
    if sections.get("validation"):
        lines += ["## Validation & status", ""]
        for v in data.get("validation", []) or ["(none)"]:
            lines.append(f"- {v}")
        hist = " → ".join(data.get("status_history", []))
        lines += ["", f"Status history: {hist}", ""]
    if sections.get("runtime"):
        lines += ["## Runtime / GPU / host", "",
                  "```", data.get("runtime", ""), "```", ""]
    if sections.get("logs"):
        lines += ["## Image & recent logs", "",
                  f"Image: `{data.get('image', '')}`", "",
                  "```", redact_secrets(data.get("logs", "")), "```", ""]
    return "\n".join(lines)
