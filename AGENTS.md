# llama_launcher: agent guide

Read this file at the start of every session. Owner habits and consent rules
live in project memory, not here. This repository is public and released
(v0.1.1): every change is user-visible, so README, CHANGELOG and packaging
change only when the work calls for it.

## What this is

Llama Launcher, a PySide6 desktop GUI that builds a llama-server command
from a profile and launches it in a container (podman or docker) or
natively, with build helpers, live monitoring, benchmarks, remote nodes and
RPC pooling across machines.

## Architecture

- src/llama_launcher/app.py: entry point, CLI (--dry-run, --profile,
  headless control), Qt application bootstrap.
- src/llama_launcher/core/: pure logic, no Qt. settings_catalog (every
  llama-server flag the app exposes, per engine), build_catalog and
  build_spec (image build options), command_builder (profile to argv),
  validation, vram and gguf (preflight from the GGUF header), capabilities
  (what the chosen engine and build support), instances, nodes, pathmap,
  router_* (router presets, models, events), lora_state, mtp_stats,
  prometheus, report.
- src/llama_launcher/services/: processes and I/O. runtime (launch, stop,
  grace period), native, terminal (emulator detection), health, metrics,
  stats and sysstat, container_stats, gpu, rpc and pool_preflight, registry
  and model_info, benchmark and benchmark_store, headless, api_key,
  router_api, lora_api.
- src/llama_launcher/store/: on-disk profiles, builds and nodes with
  atomic writes (_io).
- src/llama_launcher/ui/: main_window, panels (configure, build, monitor,
  stats, benchmark, lora), controllers (launch, monitor, benchmark,
  report), widgets, dialogs (nodes), icon.
- scripts/: install-desktop.sh (desktop entry), release-notes.sh (extracts
  a CHANGELOG section for a GitHub release).
- tests/: mirrors the package (core, services, store, ui) plus tests/guard
  for the house text rules; conftest.py at the root sets up offscreen Qt.
- docs/ is ignored and untracked; nothing under it is a source of truth.

## Commands

| Purpose | Command |
|---|---|
| Set up | `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"` (or `uv sync`) |
| Test | `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q` |
| Guards only | `.venv/bin/pytest -q tests/guard` |
| Lint | `ruff check . && ruff format --check .` (ruff on PATH at the version ci.yml pins, 0.16.5; not a project dependency) |
| Run | `.venv/bin/llama-launcher` |
| Dry run a profile | `.venv/bin/python -m llama_launcher.app --dry-run --profile NAME` |
| Local CI | `localci llama-launcher` (mirrored Qt image; keep its apt list in sync with ci.yml) |
| Release notes | `scripts/release-notes.sh <version>` |

## Gates

A cycle is not finished until, in this order: cumulative review READY,
/code-review at medium effort clean or every finding addressed, localci
green (run in the background and polled), TASKS.md and CHANGELOG.md
updated, then commit and push each behind explicit owner consent. main has
branch protection against force-push and deletion only.

Guard tests under tests/guard fail the suite on em or en dashes and
non-ASCII text outside README.md, CHANGELOG.md and RPC.md, which are
allowlisted until a documentation cycle cleans them. The doubled-hyphen
check is off (CHECK_DOUBLE_HYPHEN) while the legacy ` -- ` separators in
comments remain. UI glyphs in code are \u escapes.

`ruff check . && ruff format --check .` must be clean under the ruff
configuration the repository carries. Do not narrow it.

Preflight and catalog work is verified by running, not asserting: a
dry-run command against a real profile, or a live launch on the owner's
hardware, listed under TASKS.md "Pending owner smokes".

## Git flow

One branch per cycle named type/kebab-slug (feat, fix, chore, docs,
refactor). One cumulative commit per cycle, Conventional Commits subject with
a scope. Merge with --no-ff, delete the branch, push only after green
localci. No attribution trailers; AI-ASSISTANCE.md is the disclosure.
Releases: tag vX.Y.Z on main, CHANGELOG section is the release note source.

## Documentation routing

| Content | Goes in |
|---|---|
| Settled requirements | SPEC.md |
| Direction beyond the current cycle | ROADMAP.md |
| Current phase, open items, pending owner smokes | TASKS.md |
| User-facing changes | CHANGELOG.md [Unreleased] |
| The current cycle's plan | docs/plans/YYYY-MM-DD-topic.md (docs/ is ignored, so tracked plans need the ignore rule relaxed first) |
| History, decisions, audit records, gotchas | project memory (DevDocs/llama_launcher/memory), never code comments |
| Superpowers scratch | .superpowers/ (ignored, wiped at cycle start) |
