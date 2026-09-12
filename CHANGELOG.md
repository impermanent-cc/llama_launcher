# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each released section below is the verbatim source for that release's GitHub
notes; `scripts/release-notes.sh <version>` extracts it. Edit the entry here,
not on the release page, so the two never drift.

## [Unreleased]

Mainline llama.cpp removed `--mlock`, `--no-mmap` and `--direct-io` from its
argument parser at build 10902, so a profile carrying either of the first two
aborts the launch against a current image. Both settings move to
ik_llama.cpp, where they still work, and a profile still carrying one says so
and offers `--load-mode` instead. `--override-tensor` stops claiming to be
inapplicable on a dense model. The offload sweep no longer reports a winner
with no memory figures behind it. The flag audit that should have caught the
b10902 removals now catches the next one.

### Changed

- `--mlock` and `--no-mmap` are offered on ik_llama.cpp only. Mainline
  rejects both from build 10902, so neither reaches a mainline launch from
  the form, a profile JSON or the headless path, and `--load-mode` is the
  mainline route to the same behaviour.
- `--override-tensor` is a tuning knob on a mixture-of-experts model and
  carries no recommendation on a dense one, where it stays usable and
  unmarked. It places any tensor on any device, so no model makes it
  inapplicable.
- The `--mmproj-device` tooltip says the default follows `--device`, matching
  b10902.

### Added

- A profile carrying `--mlock` or `--no-mmap` on a llama.cpp engine warns,
  with the wording matching the route: a settings value is not sent and the
  launch runs without it, while the same spelling in the raw arguments does
  reach the launch and fails on an image from build 10902 on. A settings
  value also offers a one-click move to the matching `--load-mode`.
- The upstream flag audit reports both directions. Every flag the captured
  build accepts is either catalogued, emitted directly by the launcher, or
  listed in `tests/fixtures/unexposed_flags_mainline.txt`, and a flag on none
  of those fails the suite naming itself, so regenerating the capture is the
  audit.

### Fixed

- An ik profile carrying a leftover `--load-mode` value no longer silently
  loses `--mlock` and `--no-mmap`. The suppression is gated on the engine
  accepting `--load-mode`, which ik does not, and the two checkboxes stop
  being greyed out on an ik form, including after switching engine in place.
- The RAM shortfall wording treats a legacy `--mlock` as a lock only on an
  engine that accepts it.
- The offload sweep measures what it launches. Its own copy clears
  `--log-disable` and `--log-colors` and negates a raw `--log-jsonl`, and it
  refuses to start when the raw arguments carry `--log-disable` or a
  `--log-colors` value upstream reads as on. Each of those hides the
  load-time lines the sweep parses, and the sweep would otherwise name a
  winner with no memory figures behind it.

## [0.2.0] - 2026-09-11

The memory estimate is the headline. The Configure readout, the launch
dialog and a new `--estimate` command place every tensor of a model the way
the engine does, per card and in RAM, across `--tensor-split`,
`--n-gpu-layers`, `--n-cpu-moe`, `--n-cpu-ffn` and `--override-tensor`,
with the KV cache, the recurrent state of hybrid models, sliding-window
layers and draft models priced from the GGUF header and calibrated against
measured llama.cpp 0.4.0 and ik_llama.cpp runs. A shortfall message names
the smallest offload count or the balanced `--tensor-split` that fits. The
Benchmark tab gains an offload sweep that measures each count and writes
the fastest into the profile, the Configure tab gains a search field over
the settings and a Details section under the readout, and the catalog
tracks the llama.cpp 0.4.0 flags. `VRAM.md` describes the estimate and how
to calibrate it.

### Added

- The VRAM preflight reads the GGUF tensor table and places every tensor
  the way the engine does: per card under `--tensor-split` or free-memory
  proportions, or in RAM under `--n-gpu-layers`, `--cpu-moe`, `--n-cpu-moe`,
  `--n-cpu-ffn` and `--override-tensor`. The Configure readout shows one line
  per card and one for RAM; the launch dialog and the new
  `llama-launcher --estimate --profile NAME [--json]` show the same
  breakdown. Draft models, projectors and split models are counted. A new
  root document, `VRAM.md`, describes the estimate and how to calibrate it.
- A compute-buffer term scaled by `--ubatch-size` and flash attention, and a
  RAM check against the launch node's available memory (a warning, never a
  refusal).
- Shortfall messages name the smallest `--n-cpu-moe` or `--n-cpu-ffn` that
  fits, or the equivalent `--override-tensor` pattern where the engine lacks
  the flag; with `--fit` active the message says what llama.cpp will shrink.
- The memory estimate suggests a capacity-balanced `--tensor-split`: a card
  shortfall message names it, and `--estimate --json` carries it as
  `balanced_split`. It is a suggestion; what the estimate assumes about an
  unset split is unchanged.
- The readout's meta line carries the model's layer count, and each card
  line opens with the layer range that card holds ("layers 0 to 27 plus
  output"), or "all GPU layers, row split" under `--split-mode row`; the
  RAM line names the layers left on the host.
- A collapsible Details section under the readout with the KV cost of the
  next 1024 tokens per device, each card's average weight per layer (and the
  expert share on a mixture-of-experts model), the model's head, embedding
  and vocabulary counts, and the output tensor's size and device.
  `--estimate` prints the same block after the readout lines, and `--json`
  carries `n_layers`, per-card `layers` and `bytes_per_layer`, `kv_per_1k`
  and `output_device`.
- `--estimate` exits 6 when every card fits but RAM is over budget, with
  `ok` still true in the JSON; the README exit table carries the row.
- The estimate warns, in the readout, the launch dialog and the CLI, when a
  draft model or projector lies under no configured folder and its bytes
  are therefore not counted.
- An offload sweep in the Benchmark tab: launches the profile once per
  `--n-cpu-ffn` (dense) or `--n-cpu-moe` (MoE) count over a range the
  memory estimate prefills, benchmarks each, records the measured model, KV
  and compute buffers from the server log and shows their per-card total
  beside the estimate, marks the fastest count and writes it into the
  profile on Apply. Sweep launches on mainline llama.cpp run at log
  verbosity 4, the level at which 0.4.0 prints the buffer lines; the parser
  reads the timestamped 0.4.0 format and ik_llama.cpp's buffer lines.
- The Benchmark tab labels a stored sweep with its timestamp.
- A search field above the settings on the Configure tab: type part of a
  flag, an alias such as `-ngl`, or a group name, and the matching row
  scrolls into view and is highlighted; Enter and Shift+Enter step through
  the matches, Escape clears. Rows hidden by the current mode or engine
  never match.
- Finer `--ctx-size` presets: 1024 and the midpoints 12288, 24576, 49152,
  98304 and 196608 join the ladder.
- Four llama.cpp 0.4.0 server flags: `--kv-unified-per-slot` (per-slot context
  limit, under GPU and Memory) and `--video-fps`,
  `--video-timestamp-interval` and `--video-ffmpeg-dir` (Multimodal). All four
  are mainline-only and never reach an ik_llama.cpp launch.
- `--no-reasoning-preserve`, the control for switching reasoning preservation
  off now that llama.cpp 0.4.0 enables it by default.
- The draft offload rows accept upstream's `--override-tensor-draft`,
  `--n-cpu-moe-draft` and `--cpu-moe-draft` spellings as aliases.
- `--n-cpu-ffn` now reaches everything its MoE sibling reached: the capability
  dots (recommended on a dense model, worth tuning on a MoE one), the RPC
  centralizing warning, the benchmark run snapshot and the over-budget VRAM
  hint.
- A warning when a flag and its `--no-` twin would both act on a launch,
  naming which one llama-server will honour. Derived from the catalog, so it
  covers any such pair, and it sees a half supplied through raw args.
- A warning when `--kv-unified-per-slot` is set but the slot count is not an
  explicit number, since the KV pool size is then unknowable before launch.

### Changed

- A benchmark or sweep prompt size of 0 or below is refused the same way as
  one that is not a number.
- The suggestion dot sits directly after its setting's editor instead of
  at the row's far edge, and the `--tools` boxes lay out in two columns,
  so the settings column no longer needs a horizontal scrollbar.
- The Environment column keeps a width of 420 to 640 px and the settings
  column takes the spare width; the Name field and the profile picker in
  the top bar grow no wider than 260 and 340 px, and the spare width goes
  to the right of the bar.
- The main window's minimum width is now about 1030 px, from the Name
  field and profile picker minimums in the top bar (was about 840 px).
- The memory estimate walks a model's tensor table once per render and
  prices every balanced-split candidate and offload count from per-layer
  sums: the fit readout on a model with tens of thousands of tensors
  refreshes in tens of milliseconds instead of most of a second.
- The balanced `--tensor-split` search scores every candidate on two cards
  and climbs as many moves as there are entries on three or more.
- Card shortfall messages name the balanced split on ik_llama.cpp with
  `--fit` too, suggest no offload count where the named split already fits,
  name every card boundary, shell-quote the whole `--override-tensor`
  suggestion, and render amounts under 1 GiB in MiB.
- The router readout charges the per-card overhead once per card rather
  than once per member.
- The launch preflight reuses the Configure tab's GPU and RAM probe while
  it is fresh for the profile's node, so a launch on a remote node no
  longer freezes the window for a second round trip.
- An embedding or reranker model shows no offload recommendation dot; the
  context suggestion reads the effective context from
  `--kv-unified-per-slot` and `--parallel`.
- The sweep status line keeps "Sweep cancelled." or "Sweep failed" after
  the run ends, and a prompt size that is not a number gets its own refusal.
- The `--video-timestamp-interval` tooltip states that 0 disables the
  timestamps.
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

- Group box titles carrying an ampersand ("Model & Context" on the
  Configure tab, "CPU & ISA" on the Build tab) and the Monitor tab's
  "Enable --metrics & relaunch" button render it as an ampersand instead
  of a mnemonic underline.
- Validation's active-setting check follows the command builder's emit
  rule: a flag `--load-mode` suppresses, an enum left at its default, or a
  zero count (typed or the string "0") no longer counts as active.
- A card's output buffer line in a sweep log counts into that card's
  measured compute instead of being dropped.
- Recurrent state is no longer charged to a layer in the shared-KV tail.
- The RPC centralizing warning, the flag-pair warning and the VRAM preflight
  no longer fire for a flag the chosen engine never receives, and no longer
  treat a zero count as an active setting.
- Turning reasoning preservation off in the form while re-enabling it through
  raw args now warns instead of silently preserving.
- The deprecated marker on a settings row no longer claims every such flag was
  replaced by `--load-mode`, which was true of the load flags only.
- The `--video-ffmpeg-dir` help no longer claims the official server images
  ship without ffmpeg; they carry ffmpeg and ffprobe on PATH.
- An ik_llama.cpp profile with `--fit on` emitted `--fit on`, which ik's
  parser rejects; it now emits the bare `--fit` flag, and `off` emits
  nothing.
- The VRAM estimate charges KV only to layers that hold a cache (the
  header's full-attention interval or per-layer head counts), sizes entries
  from the header's key and value lengths, adds the recurrent state and
  context checkpoints of hybrid models, charges the logits buffer to the
  output card only, and sizes the output buffer by request slots. On
  the 2026-09-06 measurements it read up to 5 GiB high per card before.
- The sweep parser reads ik_llama.cpp's buffer lines and recurrent-state
  lines.
- The compute buffer formula gains a recurrent-activations term sized by
  the header's inner size and a per-engine scale, so a hybrid model's
  estimate on ik_llama.cpp reads inside the fit tolerance.
- The VRAM estimate prices a sliding-window model's window layers at the
  header's window head sizes and, without `--swa-full`, at the window plus
  one micro-batch per slot rather than the full context, and charges no
  cache to layers that share an earlier layer's KV. On the Gemma 4 12B and
  26B-A4B runs of 2026-09-06 the KV figure read about twice the logged size
  before and matches it exactly now; the "KV up to" label remains only for a
  header without a window pattern or on ik_llama.cpp.
- The compute buffer estimate gains a vocabulary-sized activation charged
  to each card, and the compute terms were refitted against the measured
  runs, so a card that does not hold the output layer no longer reads far
  below the buffer the server reserves there.
- Benchmark history table: prompt-eval and generation throughput now read to
  one decimal and the total to two, right-aligned, instead of printing the
  stored float at full precision.
- Memory estimate: a draft model's KV cache is sized from the layers its own
  file carries, at one sequence and at f16 unless its own cache-type
  settings say otherwise, instead of a copy of the main model's cache;
  multi-token-prediction positions are charged neither a cache nor recurrent
  state; and the state is charged once per state unit, one cell per request
  slot holding the slot's own state plus one per speculative sequence under
  a rollback spec-type (draft-mtp, draft-eagle3, draft-dflash or
  draft-dspark), whether the head speculating is a draft file or the model's
  own MTP head, with the checkpoints term left per request slot. On a 27B
  hybrid with a draft this moves the estimate onto the measured figures: the
  cache falls from twice the truth to exact, and the context checkpoints
  from 9.5 GiB to 4.7; at two slots the state matches the logged 897.75 MiB.
- Validation no longer warns that an MTP draft does not support
  `--parallel` above 1; mainline llama.cpp serves speculative decoding on
  every slot.

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

[Unreleased]: https://github.com/impermanent-cc/llama_launcher/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/impermanent-cc/llama_launcher/releases/tag/v0.2.0
[0.1.1]: https://github.com/impermanent-cc/llama_launcher/releases/tag/v0.1.1
[0.1.0]: https://github.com/impermanent-cc/llama_launcher/releases/tag/v0.1.0
