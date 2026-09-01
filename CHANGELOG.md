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

## [0.1.0] - 2026-09-01

llama_launcher v0.1.0: first release

PySide6 desktop GUI to build and launch containerized (podman/docker) or
native llama.cpp / ik_llama.cpp servers, with profiles, ~55 curated settings
plus raw args, typed mounts, mmproj/LoRA/embedding/rerank support, a router
mode, multi-node control and RPC VRAM+RAM pooling (experimental), live Monitor
(tok/s, KV, per-instance cards), benchmarks, and a headless/dry-run CLI.

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
