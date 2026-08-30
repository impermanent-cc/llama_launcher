# Security policy

## Supported versions

Llama Launcher is pre-1.0 and developed on `main`. Fixes land on `main` and ship
in the next tag; there are no backport branches.

| Version                          | Supported |
| -------------------------------- | --------- |
| `main` and the latest 0.1.x tag  | yes       |
| anything older                   | no        |

## Reporting a vulnerability

Report privately through GitHub: open the repository's **Security** tab and
choose **Report a vulnerability**. That opens a private advisory visible only to
you and the maintainer. Please do not open a public issue for a suspected
vulnerability.

This is a single-maintainer project, so triage is best effort rather than on a
schedule. Expect a first response within about a week. If two weeks pass with no
reply, feel free to open a public issue saying only that you are waiting on a
security report, with no details in it.

Please include what an attacker gains, the steps to reproduce, and the version or
commit you tested. Redact any real API keys from pasted output.

## What is in scope

Llama Launcher is a local desktop GUI that assembles and runs container, binary,
and ssh commands on your behalf. The interesting boundary is between what you
configure and what the app does with it. In scope:

- **API key leakage**: a server or router key reaching the diagnostic report,
  logs, saved profiles, benchmark snapshots, or an exported command.
- **Bypassing the pre-launch escalation screen**: getting privileged flags, host
  paths, or runtime paths into the final command without the warning that is
  meant to gate them. This covers `extra_run_args` handling, the sensitive host
  mount prefix checks, and the runtime path flag blocks.
- **File permissions**: anything writing secrets less restrictively than 0600, or
  outside the intended config directory.
- **Redaction failures** in `redact_secrets` (`core/report.py`) on realistic
  input.
- **Remote node handling**: ssh target parsing, podman over ssh command
  construction, and tunnel setup that could reach an unintended host.
- **Path handling** that escapes the directories a profile names, including mount
  and model path construction.

## What is not in scope

- **The app running a command you configured.** Llama Launcher is a launcher: if
  you add a flag, mount a directory, or bind to `0.0.0.0`, it does what you
  asked. The escalation screen exists to make that visible, not to prevent it. A
  report that the app can launch a permissive server describes expected
  behavior.
- **Upstream vulnerabilities** in llama.cpp, ik_llama.cpp, podman, docker, or the
  container images. Report those to the upstream projects. If Llama Launcher
  makes an upstream issue meaningfully worse, that part is in scope, so say so.
- **Attacks presupposing config access.** If an attacker can already write to
  your config directory, your profiles, or your shell, they can run
  `llama-server` directly.
- **AMD/ROCm code paths**, which are not implemented.
- **The Build tab.** It renders build command strings for you to run yourself and
  never executes a build.

## Hardening already in place

Worth knowing before you report: saved profiles are written 0600, the diagnostic
report and dry-run output redact API keys, exported commands are shell quoted,
sensitive host mount prefixes and runtime path flags trigger the escalation
screen, ssh runs with `BatchMode`, and GGUF header reads are size capped. If you
have found a way around one of these, that is exactly the report this policy is
for.
