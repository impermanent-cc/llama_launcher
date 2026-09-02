# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each released section below is the verbatim source for that release's GitHub
notes; `scripts/release-notes.sh <version>` extracts it. Edit the entry here,
not on the release page, so the two never drift.

## [Unreleased]

### Added

### Changed

### Fixed

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

[Unreleased]: https://github.com/impermanent-cc/llama_launcher/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/impermanent-cc/llama_launcher/releases/tag/v0.1.0
