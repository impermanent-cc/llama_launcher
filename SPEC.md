# llama_launcher: specification

Settled requirements. Each numbered item is a decision the owner has agreed
to; grilling sessions append or amend items here before a plan is written.
Items are present tense and testable. Started fresh at the workflow stamp
(2026-09-02); earlier design records stay in project memory and are not
carried into this file.

## 1. Purpose

1.1 Llama Launcher turns a saved profile into a correct llama-server
command and runs it, in a container or natively, on one machine or across
remote nodes, with the state of the running server visible in the GUI.

## 2. Requirements

2.1 A profile is the unit of configuration: engine (mainline llama.cpp or
ik_llama.cpp), image or native binary, model path, GPU mode, and any
catalogued llama-server setting. `--dry-run --profile NAME` prints the
exact command without launching.

2.2 The settings catalog exposes only flags the chosen engine accepts;
mainline-only flags never reach an ik launch, and options only ik defines
never reach a mainline build command. Re-audits against upstream
`common/arg.cpp` and the CMake option lists are periodic maintenance,
carried out by hand: the flag fixtures are the suite's only upstream
oracle, and the build catalog has none.

2.3 Before a launch the app reads the GGUF header, including its tensor
table, and estimates memory per GPU and for host RAM against the profile's
placement flags and context size, and warns or refuses according to
validation, rather than letting the server fail late. Items 2.14 to 2.25
define the estimate and its messages.

2.4 Launch modes: container via podman or docker, native binary,
foreground in a detected terminal emulator, or headless control. The stop
grace period is configurable.

2.5 Remote nodes are reached over ssh and require podman on the node. An
RPC pool spreads a model across nodes' rpc-server workers; the RPC workers
table shows node, device and contribution. Pooled inference across
CPU-only workers is experimental and known to crash upstream.

2.6 The Monitor tab shows every instance as a card with logs and live
stats; the Stats dock shows CPU, GPU and memory; the Benchmark panel runs
prompt and generation sweeps and shows deltas against the previous run.
Each run records a config snapshot of the settings that shape throughput,
including both CPU offload counts.
The Benchmark panel also runs the offload sweep of 2.26 to 2.31.

2.7 The router API key, LoRA adapter scales and model switching are driven
through the server's HTTP API, never by restarting the server.

2.8 One default port lives in core.spec.DEFAULT_PORT; no literal port
numbers elsewhere.

2.9 A setting whose behaviour upstream now enables by default, or that
upstream has superseded, stays catalogued and marked deprecated so an
older image keeps the control, and the flag a current build needs for the
opposite choice is catalogued alongside it.

2.10 Setting both a catalogued flag and its catalogued `--no-` twin raises
a validation warning, for any such pair rather than a named one: upstream
resolves the pair by argv order, so the launch would otherwise choose
silently.

2.11 With `--kv-unified-per-slot N` set and no `ctx-size`, on an engine that
accepts the flag, the VRAM preflight takes the effective context as
`parallel * N` where `--parallel` is an explicit positive number, and warns
that the KV pool size is unknown where `--parallel` is auto.

2.12 Capability relevance follows upstream's own split of a flag pair, which
is not symmetric: `--n-cpu-moe` is recommended on a MoE model and
inapplicable on a dense one, which has no experts for it to move;
`--n-cpu-ffn` is recommended on a dense model and a tuning knob on a MoE
one, whose non-expert layers still carry dense FFN weights.

2.13 Every catalogued flag that centralizes memory on the head
(`--cpu-moe`, `--n-cpu-moe`, `--n-cpu-ffn`, `--no-kv-offload`,
`--override-tensor`) warns on an RPC pool launch, for those of them that
reach argv for that launch's engine.

2.14 The GGUF reader returns, beside the hyperparameters, the tensor table
of the file: each tensor's name, element count, GGML type and byte size,
plus the FFN width, the vocabulary size and the experts used per token. A
split model, whose parts share a directory and carry part-of-total names,
yields the tables of every part. When the table cannot be read the
estimate counts the file size as GPU weights, as before.

2.15 A pure placement function in core assigns every tensor to a card or
to host RAM by applying, in this order, the rules the engine applies:
`--n-gpu-layers N` places the output layer and the last N minus one block
layers on cards under llama.cpp, and the last N block layers with the
output layer only past them under ik_llama.cpp, every other layer staying
in RAM; `--cpu-moe`, `--n-cpu-moe` and
`--n-cpu-ffn` expand to upstream's per-layer patterns, using upstream's
expert and dense FFN regexes verbatim; `--override-tensor` patterns are
evaluated last, against every tensor no CPU rule claimed, including tensors
on layers `--n-gpu-layers` keeps in RAM; they are matched in order, first
match wins, a CPU target means RAM and any other target means a card. A flag participates only when
the chosen engine accepts it, through the same gate the command builder
uses. Under ik_llama.cpp the `auto` value of n-gpu-layers means no layer
is offloaded, matching what the command sends. Token embeddings stay in
RAM as the input layer; a model without an `output.weight` tensor reuses
them as its output matrix, and that copy is placed with the output layer.

2.16 GPU layers are distributed across cards by `--tensor-split` when set
and otherwise in proportion to each card's free VRAM, as both engines do.
A layer's KV cache is placed with its layer, and `--no-kv-offload` moves
all of it to RAM. Split mode `row` places weights by the same proportion
and the KV cache and compute buffer on `--main-gpu`; split mode `none`
places everything on `--main-gpu`; ik's `graph` mode is placed like
`layer`.

2.17 A draft model is placed by the same function with the draft twins of
the offload flags, and carries its own KV cache at the draft context when
set and the main context otherwise. The draft twins are the catalog's
existing mainline-only rows `spec-draft-override-tensor`,
`spec-draft-n-cpu-moe` and `spec-draft-cpu-moe`, which also accept
upstream's `--override-tensor-draft`, `--n-cpu-moe-draft` and
`--cpu-moe-draft` spellings as aliases. A projector file adds its size to
the GPU unless `--no-mmproj-offload` is set.

2.18 The KV cache is charged only to layers that hold one. A layer holds
a KV cache when the header's full-attention interval names it (layer i,
counting from zero, when i plus one is a multiple of the interval) or when
a per-layer KV head-count array gives it a non-zero count; a model whose
header carries neither charges every layer. The cache type settings and
the header's key and value lengths (embedding size over head count when
absent) size each entry. A sliding-window model is estimated at full context on its KV
layers and labelled as an upper bound wherever it is shown; a hybrid model
without a sliding window carries no label.

2.19 The compute buffer is estimated per card from a formula of ubatch-
scaled terms (FFN activations, attention scores, residual stream, recurrent
activations sized by the header's inner size) plus a fixed backend constant, with the attention-scores term absent when flash
attention is on or auto, and the card and host figures scaled by one
constant per engine, fitted on that engine's calibration records. The ubatch-scaled logits term is charged only to
the card the placement gives the output tensor, or to RAM when that tensor
stays there. RAM also carries a host compute buffer, a multiplier from the
same table on the card formula without the logits term, and an output
buffer of vocabulary size times four
bytes times the slot count (the `--parallel` setting, else the engine's
default, four on llama.cpp and one on ik_llama.cpp). The term table lives in the VRAM module and every
rendering of the value marks it approximate. The multipliers are fitted
against the measured breakdowns kept with the project's calibration
records: KV plus recurrent state reads high by at most a quarter and never
low on the sum across cards, and per card within one layer's KV of that;
compute and host buffers never read low and read at most two and a half
times high, compared per card as sorted figures since the output card
follows settings the records do not carry. The calibration procedure
against llama.cpp's memory breakdown is documented, per engine, so the
constants can be refitted when upstream changes its graphs.

2.20 The fit check compares each card's estimated total (weights, KV,
compute, overhead) with that card's free VRAM, and the RAM total
(weights, KV, host buffers) with the available memory of the launch node,
local or over ssh. A card shortfall names the card and the missing amount.
A RAM shortfall is a warning and never a refusal; under load-mode `none`
or `mlock` its wording states the launch fails, otherwise that the server
pages.

2.21 A card shortfall message names the smallest `--n-cpu-moe` value (MoE
model) or `--n-cpu-ffn` value (dense model) at which every card fits,
found by evaluating the placement function for increasing counts. On an
engine that lacks `--n-cpu-ffn` the message gives the equivalent
`--override-tensor` value, an explicit alternation of layer indices with
upstream's dense FFN regex. When no count fits, the message says so.

2.22 On mainline, `--fit` acts only on values left unset: it is active when
`--fit` is unset or on and none of `--n-gpu-layers`, `--override-tensor`,
`--cpu-moe`, `--n-cpu-moe`, `--n-cpu-ffn` or `--tensor-split` is set and
the split mode is `layer` (or there is one card). Under an active fit a
shortfall produces a note that llama.cpp will not fail: with no
`--ctx-size` it shrinks the context toward `--fit-ctx` (default 4096) and
then drops whole layers, and the note names the context it reaches,
solved against the summed free VRAM less the per-card `--fit-target`
margin (default 1024 MiB); with `--ctx-size` set it keeps that context and
moves whole layers to RAM, experts first on a MoE model, and the note says
so without a predicted context. The note reaches the abortable launch
dialog only when `--fit` is unset; an explicit `on` keeps the launch
silent and the readout carries the note. Unset is treated as active on
every image.

2.23 On ik_llama.cpp with `--fit` on, a MoE model that does not fit gets a
note that ik keeps the experts of as many layers as needed in RAM with
context and layer count unchanged, and the RAM total includes them; a dense
model that does not fit gets a warning that ik refuses to load it. With
`--fit` unset or off on ik the plain shortfall applies. The command builder
sends bare `--fit` for `on` under ik and nothing for `off`, ik's default,
and the setting's tooltip states each engine's default.

2.24 The Configure readout shows the model meta line, one line per card
("GPUn est / free GiB" with the weights, KV and compute parts in
brackets), and one RAM line, with word wrap on so no line widens the
window; the label's tooltip carries the full part list with the
approximate and upper-bound notes. The launch preflight dialog shows the
same breakdown. The router readout sums per-member card totals and the
pool fit consumes the same breakdown with unchanged semantics. One pure
core function renders these lines for the readout, the tooltip, the
dialog and the CLI.

2.25 `--estimate --profile NAME` prints the breakdown of 2.24 for the
profile's launch node, and with `--json` prints it as JSON.

2.26 An offload sweep launches the current profile once per count as a
detached container named `llama-<slug>-sweep`, waits for `/health` up to a
ready timeout (panel field, default 600 seconds), reads the memory lines
from the container log, runs the benchmark with the panel's current prompt
sizes, n-predict, warmup and repeats, stops and removes the container, and
proceeds to the next count; on mainline llama.cpp the launch carries log
verbosity 4 when the profile's is lower, on other engines the profile's
own. It refuses, with a one-line message in the panel, a native, RPC, router or remote-node profile, a profile with no
model, an engine that does not accept the sweep's knob, raw arguments that
carry the knob's flag, an empty prompt-sizes field, and a start while the
profile's own instance, a previous sweep container, a benchmark or another
sweep is running. A cancelled sweep is shown but not stored; the stored
sweep of the loaded profile is shown when the profile loads.

2.27 The sweep varies `--n-cpu-ffn` on a dense model and `--n-cpu-moe` on
a MoE one, over the counts from `from` to `to` inclusive in steps of
`step`, three panel fields prefilled when the profile or its estimate
changes: `from` is the smallest fitting count of 2.21, or 0 when every card
fits or no count fits; `to` is `from` plus 8; `step` is 2. Each point's
settings are the profile's settings with that count laid over.

2.28 A point whose container fails to start or is not ready within the
timeout is recorded as failed with the last line of its log, its container
stopped and removed, and the sweep continues with the next count. Cancel
ends the sweep after stopping and removing the current point's container.

2.29 Each ready point records the seconds from launch to ready and the
load-time memory lines llama-server logs: the model, KV, recurrent-state
and compute buffer sizes per device and the output buffer. A device named
`CUDA<n>` maps to card `n`; every other device counts toward RAM. A line
may carry a timestamp and level prefix; a model line without a kind word
(ik_llama.cpp) counts as model; a recurrent-state (RS) line counts into the
KV figure.

2.30 The latest sweep of a profile is stored in its own file beside the
benchmark history, which the sweep never touches: the knob, the counts, and
per point the status, the ready seconds, the benchmark rows, the measured
memory and the estimate's per-card model plus KV plus compute and its
RAM total for that count. The winner is the ok point with the highest generation tokens per second at the
largest prompt size.

2.31 The Benchmark tab carries a sweep control row (knob label, from, to,
step, ready timeout, Run sweep or Cancel, Apply) and a sweep table under
the run table in a splitter: count, status with the failure line in its
tooltip, ready seconds, prompt and generation tokens per second at the
largest prompt size, per card measured against estimated GiB for model plus
KV (recurrent state included) plus compute, without the per-card overhead
and checkpoint terms the log never reports, and RAM measured against
estimated; the winning row is marked. Apply writes the winner's count into the Configure form and saves
the profile. The compute constants of 2.19 are not changed by a sweep.

2.32 A recurrent layer (a layer without a KV cache, and without a dense
feed-forward width where the header carries per-layer widths, on a model
whose header carries recurrent-state sizes; every layer on a model with
state sizes and no attention heads) adds its state in f32, the convolution
state (kernel size minus one, times inner size plus twice the group count
times state size) plus the state matrix (state size times inner size),
once per request slot, to the card that holds the layer, or to RAM where KV
would go. A checkpoints term, the `--ctx-checkpoints` setting (default 32)
times the recurrent state of every recurrent layer per slot, is charged to
RAM, where the server keeps its checkpoints. Both appear in every readout
total, the dialog and `--estimate --json` (card key `state`, RAM keys
`state` and `checkpoints`), and the sweep's estimated figures of 2.31
include the state and exclude the checkpoints. A per-layer KV head-count
array sizes each layer's cache by its own count.

## 3. Constraints

3.1 Python 3.12 and 3.13 are the tested floor and ceiling; the code needs
only 3.11 (enum.StrEnum, datetime.UTC) but the declared floor is the
CI-backed one.

3.2 No host paths, emails or secrets in tracked files; the repository is
public.

3.3 Qt runtime libraries used headless are listed in ci.yml and mirrored
in the localci Containerfile; the two lists stay in sync.

3.4 Tracked text is ASCII and free of em and en dashes except README.md,
CHANGELOG.md and RPC.md until their cleanup cycle. UI glyphs in code are
\u escapes.

3.5 The placement regexes for `--cpu-moe`, `--n-cpu-moe`, `--n-cpu-ffn`
and fit's all-experts pattern are pinned by a fixture to upstream's
strings, maintained by hand like the flag fixtures.

3.6 ruff, at the version the workflow standard pins in its CI lint job,
reports nothing for `ruff check .` and `ruff format --check .` under the
[tool.ruff] configuration in pyproject.toml, which is the standard's block
(line-length 88, rule sets E, F, W, I, UP, B and RUF, E501 ignored, *.md
excluded) and is not narrowed here. CI runs both as a lint job next to the
sanity job.

## 4. Out of scope

- ROCm and AMD GPUs until a contributor with the hardware wires them up.
- HF download flags (--hf-repo, --hf-file, --model-url and kin): they cut
  against the mount-a-local-path model and bypass the VRAM preflight.
- Two-token flags the catalog cannot express (--spec-replace,
  --control-vector-layer-range) until they get panel plumbing like LoRA.
- Reading a sliding-window model's per-layer window pattern; the KV
  estimate for such models is the labelled upper bound of 2.18.
- Multi-head latent attention caches (deepseek2 and kin): the estimate
  prices them at the header's key and value lengths per head, far above
  the latent cache the engine keeps, and labels nothing.
- The CPU_REPACK copy of weights run on the CPU (a second, repacked copy
  beside the mapped one on a CPU-only launch); the RAM estimate does not
  model it.
- Reproducing mainline fit's per-card layer distribution; the predicted
  context of 2.22 is solved against the summed budget.
- A ubatch sweep, a headless `--sweep` command and sweeps on a remote
  node: ROADMAP Later.
- Synthetic speculative acceptance flags (--spec-synth-len,
  --spec-synth-rates): upstream marks them benchmarking only and they
  falsify acceptance, so a profile carrying them serves nonsense.
