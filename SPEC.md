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
one, whose non-expert layers still carry dense FFN weights. A model the
capabilities mark as an embedding model (an embedding architecture or a
header pooling type) gets no recommendation on either knob; both stay
usable.

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
header carries neither charges every layer. The last `shared_kv_layers`
layers of a model whose header carries that key own no cache. The cache
type settings and the header's key and value lengths (embedding size over
head count when absent) size each entry; a window layer, one whose entry
in the header's `sliding_window_pattern` array is true, is sized by the
header's `key_length_swa` and `value_length_swa` instead. On mainline
llama.cpp without `--swa-full`, a window layer holds per request slot the
smaller of the per-slot context and the sliding window plus the micro-batch
size, rounded up to a multiple of 256 tokens (with `--kv-unified`, one
stream holding the smaller of the context and the window times the slot
count plus the micro-batch size); with `--swa-full`, and on ik_llama.cpp
always, it holds the full context. A draft model follows the same rules.
A model whose header carries a sliding window but no pattern array is
estimated at full context on every KV layer; that case, and a
sliding-window model on ik_llama.cpp, is labelled as an upper bound
wherever the KV figure is shown (readout label, message and the JSON key
`kv_upper_bound`); no other model carries the label.

2.19 The compute buffer is estimated per card from a formula of ubatch-
scaled terms (FFN activations, attention scores, residual stream, recurrent
activations sized by the header's inner size, and a vocabulary-sized
activation the card graph reserves) plus a fixed backend constant, with the attention-scores term absent when flash
attention is on or auto, and the card and host figures scaled by one
constant per engine, fitted on that engine's calibration records. The ubatch-scaled logits term is charged only to
the card the placement gives the output tensor, or to RAM when that tensor
stays there. RAM also carries a host compute buffer, a multiplier from the
same table on the card formula without the logits or the vocabulary term,
since the host graph reserves neither, and an output
buffer of vocabulary size times four
bytes times the slot count (the `--parallel` setting, else the engine's
default, four on llama.cpp and one on ik_llama.cpp). The term table lives in the VRAM module and every
rendering of the value marks it approximate. The multipliers are fitted
against the measured breakdowns kept with the project's calibration
records: KV plus recurrent state reads high by at most a quarter and never
low on the sum across cards (on the RAM figure for a record with no card
figures), and per card within one layer's KV of that;
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
upstream's dense FFN regex, rendered in single quotes as the whole value
the search evaluated, the profile's own rules followed by the alternation,
so a paste reproduces the searched state. When no count fits, the message
says so. A shortfall amount under 1 GiB renders in MiB, never as "~0.0
GiB".

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
("GPUn est / free GiB" with the layer range, weights, KV and compute parts
in brackets), and one RAM line, with word wrap on so no line widens the
window; the label's tooltip carries the full part list with the approximate
and upper-bound notes. The meta line carries the model's total layer count
("48 layers") after the size label. Each card line opens its bracket with
the layer range the card holds under the profile's `--n-gpu-layers` and
split ("layers 0 to 27"), followed by "plus output" on the card that holds
the output layer; the RAM line names the layers left on the host and "plus
output" when the output layer stays there. A layer's home is the card the
engine assigns it to under `--n-gpu-layers` and the split, so an
`--override-tensor` rule that promotes part of a host layer's weights onto a
card leaves it in the host range, and `--no-kv-offload` moves the cache
without moving the layer. Split mode `row` lists the offloaded layers
followed by "row split" on every card line, since every card holds a share
of each. The ranges cover the main model; a draft model's or projector's
bytes sit in the weights figure without a range. The launch preflight
dialog shows the same breakdown.
The router readout charges each card's per-card overhead once and sums
weights, KV and compute across members; the pool fit consumes the same
breakdown with unchanged semantics. One pure core function renders these
lines for the readout, the tooltip, the dialog and the CLI.

2.25 `--estimate --profile NAME` prints the breakdown of 2.24 for the
profile's launch node followed by the details block of 2.41, and with
`--json` prints both as JSON. It exits 0 when
every card fits and RAM fits or is unknown, 6 when every card fits and RAM
is over budget, and 3 when any card is over budget whatever RAM does; the
JSON `ok` is true whenever every card fits, since a RAM shortfall is a
warning by 2.20. README's exit table lists the codes.

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
RAM total for that count, and the timestamp taken when the sweep started.
The winner is the ok point with the highest generation tokens per second at
the largest prompt size.

2.31 The Benchmark tab carries a sweep control row (knob label, from, to,
step, ready timeout, Run sweep or Cancel, Apply) and a sweep table under
the run table in a splitter: count, status with the failure line in its
tooltip, ready seconds, prompt and generation tokens per second at the
largest prompt size, per card measured against estimated GiB for model plus
KV (recurrent state included) plus compute, without the per-card overhead
and checkpoint terms the log never reports, and RAM measured against
estimated; the winning row is marked. Apply writes the winner's count into
the Configure form and saves the profile. Selecting a profile loads its
stored sweep of 2.30 into the table, labelled with that sweep's timestamp;
a profile without one shows an empty table and no label. The compute constants of 2.19
are not changed by a sweep.

2.32 A recurrent layer (a layer the attention rules of 2.18 leave uncached,
before the shared-KV tail is cleared, and without a dense
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

2.33 A balanced `--tensor-split` is computed whenever the estimate is
knowable, at least two cards are visible, at least as many layers are placed
on cards as there are cards, and the split mode places layers in contiguous
runs, which by 2.16 is every mode but `row` and `none`. The candidates are
the whole-layer boundaries that leave no visible card empty, and a candidate
is feasible when every card fits at the profile's context. A candidate
scores as its feasibility first. Among candidates where every card fits, the
next term is the minimum per-card capacity, a card's margin (its free VRAM
less its estimated total) divided by its marginal cost per token, taking the
minimum over cards whose marginal cost is non-zero, and then the minimum
margin, which decides where no card has a non-zero marginal cost. A
candidate that leaves any card over budget scores on its minimum margin
alone, since a negative margin over a cost is not a capacity and ranking it
as one would favour the split with the deeper shortfall. On two cards every
candidate is scored and the best-scoring one is returned. On three or more
cards the search starts at the split the profile itself would use, scores
every candidate one layer away across any boundary, takes the best of those
while one scores strictly better than the current candidate, and stops when
none does or after as many moves as there are entries placed on cards; on a curve with
more than one peak it can stop at a lower peak. A card's marginal cost per
token is measured at
the candidate's own split: the difference between its KV and state priced at
the profile's context plus 4096 tokens and the same figure at that context,
divided by the step, so the window caps, cache types and slot rules of 2.18,
2.19 and 2.32 need no second formula. The value is rendered as layer counts
per card (`38,28`), whose boundary cannot round onto another layer, and a
message sentence naming it names every card boundary.

2.34 The balanced split of 2.33 is a suggestion and never an assumption:
with `--tensor-split` unset the distribution of 2.16 stays the engines'
free-VRAM proportion. `--estimate --json` carries it as `balanced_split`
with the keys `value`, `layers_per_card`, `boundary_layers` and `fits`, the
last being whether the split alone fits with no offload; the key is absent
whenever 2.33 computes no split. When every card fits, the readout and its
messages are unchanged and the split is carried in the JSON alone. A card
shortfall message, on either engine and whether or not ik keeps experts in
RAM by 2.23, names the balanced split only where applying it would be an
improvement: it places layers differently from the split the profile
already resolves to, and it either fits on its own or needs a smaller
offload count than the profile's own split does. Where it is named, the
message states the profile's current `--tensor-split` value if one is set
and that the suggestion replaces it, and, where `--fit` would otherwise act
by 2.22, states that an applied split keeps it from acting. When the
balanced split does not fit on its own, the offload count search of 2.21
runs once at that split as well as at the profile's, without re-balancing
per count; the message names the count and the split together when the
balanced split is named, and names the profile's own count otherwise. Where
the named balanced split fits on its own, the message carries no offload
count clause at all.

2.35 A draft model or projector whose path lies under no configured folder
counts as zero bytes in the estimate and produces a dialog-level message
naming the setting and the path and stating that its bytes are not
counted, in the readout, the launch dialog and the CLI.

2.36 The launch preflight reuses the Configure panel's GPU and RAM probe
when the cached result is younger than the panel's TTL, and probes afresh
only otherwise; a launch never repeats a probe the readout has just made.

2.37 The Configure tab carries a search field above the settings scroll
area on its right half, with a match counter reading "n of m" or "no
match". The typed text matches as a case-insensitive substring against
every setting's flag, its aliases and every group title, ignoring a typed
leading dash; rows hidden for the current mode or engine never match.
Matches are ordered top to bottom as laid out. The first match scrolls into
view and its row label is tinted while the text still matches it; a group
match scrolls the group box to the top and tints its title. Enter moves to
the next match and Shift+Enter to the previous, Escape clears the field,
and keyboard focus never moves into a setting widget. Clearing the field or
changing the text removes the tint.

2.38 Every setting row places its suggestion dot directly after the
editor, 16 px wide, with the row's stretch after the dot; the dot stays
hidden while it has nothing to say. The settings column's minimum width,
measured offscreen with every group visible, stays under 600 px.

2.39 A multiselect setting lays its boxes out in two columns, filled row by
row with "all" first, so `--tools` shows four rows of two.

2.40 The `--ctx-size` preset list is 0, 1024, 2048, 4096, 8192, 12288,
16384, 24576, 32768, 49152, 65536, 98304, 131072, 196608 and 262144, in
that order, and the combo stays editable.

2.41 A collapsible "Details" section sits under the readout lines of 2.24,
collapsed on every start, and holds, as plain lines: the KV cost per 1024
tokens as the marginal cost at the profile's context (the estimate at
context plus 1024 less the estimate at context), in total and per device;
per card, the average weight bytes per layer it holds, with the expert
share of one layer on a mixture-of-experts model; the header facts
(attention heads, KV heads, embedding width, vocabulary and sliding window
size when present); and the output tensor's size and device. The meta
label tooltip and the launch preflight dialog carry none of it. The
`--estimate --json` output carries the layer count, each card's layer
range, the KV cost per 1024 tokens, the per-layer weights and the output
device under the same names as the readout uses.

2.42 The Environment column is bounded to a width of 420 to 640 px and the
settings column absorbs the spare width; the fields inside the column keep
filling it. In the top bar the Name edit is bounded to 160 to 260 px and
the profile combo to 200 to 340 px, the buttons follow the combo, a stretch
follows the buttons and the status label sits at the right edge. At a
window width of 1600 px neither field passes its maximum and the
Environment column stays inside its bounds.

2.43 The Benchmark tab's run-history table renders `pp t/s` and `gen t/s` to
one decimal and `total s` to two, reading the values the run stores at full
precision. The `size` and `prompt_n` cells render the stored value's own
text, so a count carries no decimals and no thousands separator. The cells of
all five columns are right-aligned; the column headers and the run's group
header row keep their own alignment. A stored value that is not a number
renders as its own text, and a missing one renders an empty cell. The stored
benchmark file, the sweep table of 2.31 and the delta summary line keep the
precision they already have.

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

3.7 One debounced Configure render, or one `--estimate` run, walks the
tensor table once: per-layer, per-device byte sums are computed once per
tensor table and settings, and every balanced-split candidate and offload
search count is priced from those sums. On an engine without `--n-cpu-ffn`
the offload search's appended `--override-tensor` entry is priced from a
second set of sums, so such a report costs two walks and never more. The
expert-share scan behind `expert_layer_bytes` is a second pass over the
table, memoised per table, so a render after the first one pays no extra
walk for it.

## 4. Out of scope

- ROCm and AMD GPUs until a contributor with the hardware wires them up.
- HF download flags (--hf-repo, --hf-file, --model-url and kin): they cut
  against the mount-a-local-path model and bypass the VRAM preflight.
- Two-token flags the catalog cannot express (--spec-replace,
  --control-vector-layer-range) until they get panel plumbing like LoRA.
- A sliding-window model whose header carries no `sliding_window_pattern`
  array (the pattern is hardcoded upstream); its KV estimate is the
  labelled upper bound of 2.18.
- Balancing finer than a whole layer: moving tensor families across the
  split boundary with `--override-tensor`, which needs measured probes
  rather than the static estimate, and any balanced split for split modes
  `row` and `none`, which have no layer boundary to move.
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
- Remembering the Details section of 2.41 open or closed across runs;
  it starts collapsed.
- Synthetic speculative acceptance flags (--spec-synth-len,
  --spec-synth-rates): upstream marks them benchmarking only and they
  falsify acceptance, so a profile carrying them serves nonsense.
