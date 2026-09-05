# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each released section below is the verbatim source for that release's GitHub
notes; `scripts/release-notes.sh <version>` extracts it. Edit the entry here,
not on the release page, so the two never drift.

## [Unreleased]

### Added

- Four llama.cpp 0.4.0 server flags: `--kv-unified-per-slot` (per-slot context
  limit, under GPU and Memory) and `--video-fps`,
  `--video-timestamp-interval` and `--video-ffmpeg-dir` (Multimodal). All four
  are mainline-only and never reach an ik_llama.cpp launch.
- `--no-reasoning-preserve`, the control for switching reasoning preservation
  off now that llama.cpp 0.4.0 enables it by default.
- A warning when a flag and its `--no-` twin would both act on a launch,
  naming which one llama-server will honour. Derived from the catalog, so it
  covers any such pair, and it sees a half supplied through raw args.
- A warning when `--kv-unified-per-slot` is set but the slot count is not an
  explicit number, since the KV pool size is then unknowable before launch.
- `--n-cpu-ffn` now reaches everything its MoE sibling reached: the capability
  dots (recommended on a dense model, worth tuning on a MoE one), the RPC
  centralizing warning, the benchmark run snapshot and the over-budget VRAM
  hint.

### Changed

- `--reasoning-preserve` is marked deprecated: llama.cpp 0.4.0 preserves the
  reasoning trace by default, so the flag only changes behaviour on an older
  image. Saved profiles keep their meaning; the key was not repurposed.
- The VRAM preflight sizes the KV cache from `--kv-unified-per-slot` times the
  slot count when no context size is set, so the estimate follows the pool the
  server will actually allocate. Every single-node estimate path now shares
  that one rule.
- `GGML_CUDA_PEER_MAX_BATCH_SIZE` is offered for ik_llama.cpp builds only.
  Mainline ggml no longer defines it, so a mainline build command no longer
  carries a variable CMake ignores.

### Fixed

- The RPC centralizing warning, the flag-pair warning and the VRAM preflight
  no longer fire for a flag the chosen engine never receives, and no longer
  treat a zero count as an active setting.
- Turning reasoning preservation off in the form while re-enabling it through
  raw args now warns instead of silently preserving.
- The deprecated marker on a settings row no longer claims every such flag was
  replaced by `--load-mode`, which was true of the load flags only.
- The `--video-ffmpeg-dir` help no longer claims the official server images
  ship without ffmpeg; they carry ffmpeg and ffprobe on PATH.

## [0.1.1] - 2026-09-03

### Added

### Changed

- Lint and format gate: `ruff check` and `ruff format --check` run in CI at a
  pinned ruff version under the `[tool.ruff]` block in pyproject.toml; the
  whole tree is reformatted and the findings fixed. No behaviour change.
- README: new Benchmark and Fetch latest screenshots; refreshed Build, Monitor
  and Router screenshots.
- Code comments and docstrings now describe only what the code does; history,
  review and plan narration was removed throughout `src/` and `tests/`.

### Fixed

- Dialog titles no longer carry a Qt-added " — Llama Launcher" suffix.
- The Configure form hides the native-only and RPC-only rows on first start
  instead of only after the launch-mode combo is flipped.

## [0.1.0] - 2026-09-02

llama_launcher v0.1.0: first release

PySide6 desktop GUI to build and launch containerized (podman/docker) or
native llama.cpp / ik_llama.cpp servers, with profiles, a curated settings
catalog plus raw args, typed mounts, mmproj/LoRA/embedding/rerank support, a
router mode, multi-node control and RPC VRAM+RAM pooling (experimental), live
Monitor (tok/s, KV, per-instance cards), benchmarks, and a headless/dry-run CLI.

The settings catalog covers 253 flags, audited in both directions against
llama.cpp b10711 and against ik_llama.cpp, with each engine's accepted-flag
list pinned by a test so an upstream rename fails a test rather than a launch.
Flags only mainline accepts are tagged as such and never reach an ik launch,
and ik_llama.cpp's own surface is now covered too: an ik profile reaches 183
settings where it previously reached 109. That includes ik's MoE expert
placement and prefetch, multi-GPU graph split and exchange precision, extra
per-layer KV-cache types, context checkpoints, and its embedding output
formats. Five settings that mainline had renamed now use the spelling both
engines accept, so they work on ik instead of being mainline-only; their old
spellings stay as aliases so raw args written against them still fold onto the
setting.

Divergence bugs that fell out of that audit and are fixed: a draft model on an
ik profile emitted mainline's --spec-draft-model, which ik rejects outright, so
the launch died; router mode was selectable with the ik engine (or an ik image)
even though ik has no router at all, which is now refused with an explanatory
error, as is a router member whose engine differs from the router's; and the
"auto"/"all" layer-count tokens, which ik's parser rejects, are translated on an
ik launch instead of killing it. A raw --logit-bias now adds to the form's
entry rather than replacing it.

LoRA adapters can be rescaled on a running server through /lora-adapters
without a restart.

A Build helper tab generates copyable build commands for compiling either
engine from source (a native cmake configure/build pair, or a Containerfile
plus its podman build command) from a source-verified CMake option catalog.
It never runs a build itself. Saved build configs and an Outputs table track
each resulting image or binary as built, missing, or untracked, with guarded
delete and a use-in-profile action.

Docker and podman paths, single-server plus router modes, and the embedding
and reranking (RAG) path validated live. Known gap: AMD/ROCm GPUs untested.

[Unreleased]: https://github.com/impermanent-cc/llama_launcher/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/impermanent-cc/llama_launcher/releases/tag/v0.1.1
[0.1.0]: https://github.com/impermanent-cc/llama_launcher/releases/tag/v0.1.0
