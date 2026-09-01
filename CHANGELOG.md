# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each released section below is the verbatim source for that release's GitHub
notes; `scripts/release-notes.sh <version>` extracts it. Edit the entry here,
not on the release page, so the two never drift.

## [Unreleased]

### Added

- 80 further `llama-server` settings, from an audit of the Configure catalog
  against `llama-server --help` on build b10711. New groups for Context
  Extension (RoPE and YaRN), CPU & Threading, Logging and Multimodal; new
  entries elsewhere for `--alias`, TLS (`--ssl-key-file` / `--ssl-cert-file`),
  `--api-prefix`, `--timeout`, device memory auto-fit (`--fit`), the ngram
  speculative-decoding tuning knobs, and more.
- Live LoRA scale control. The LoRA section can now read the scales a running
  server is using and push new ones to it through `/lora-adapters`, so an
  adapter can be A/B'd against the base model without a restart. Pairs with the
  new `--lora-init-without-apply` setting, which loads adapters inactive.
- A flag-acceptance guard for both engines: `tests/fixtures/regen_flags.sh`,
  `regen_ik_flags.sh` and a test that every catalog flag is one the engine it
  reaches actually accepts, so a rename or a fork divergence cannot pass
  unnoticed again. The ik fixture is built by executing each candidate, not by
  reading `--help`, because ik's help under-reports its own parser.

### Changed

- The default port lives in one place (`core.spec.DEFAULT_PORT`) instead of
  being repeated as a literal at 33 call sites. Upstream has announced a future
  move from 8080 to 9931 (ggml-org/llama.cpp#26508) but has not made it; when it
  lands, this is a one-line change. Nothing about a launch changes today, since
  the launcher has always passed `--port` explicitly.

### Fixed

- The "on-demand tensor reading" setting emitted `--tensor-read-lazy`, which
  upstream renamed to `--lazy-mode`. Any value other than the default `auto`
  therefore failed the launch on current builds. The saved-profile key is
  unchanged, so existing profiles keep their value.
- 75 settings that only mainline llama.cpp accepts were reaching ik_llama.cpp
  launches, where setting any one of them failed with "unknown argument". They
  are now tagged mainline-only and are dropped from ik launches and hidden on
  an ik profile's form. 27 of these predate this release; the rest arrived with
  the catalog additions above. Existing ik profiles keep the stored values, they
  simply stop being passed to the server.

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
