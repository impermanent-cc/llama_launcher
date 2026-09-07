# VRAM and RAM: how the memory estimate is built

Llama Launcher's memory estimate reads a model's GGUF tensor table and
places every tensor the way llama.cpp and ik_llama.cpp place it, so the
Configure readout, the launch preflight dialog and `llama-launcher
--estimate` all show the same numbers for the same profile. This document
describes what the estimate counts, what its messages mean, how to
recalibrate its fitted constants, its JSON keys, and where it is known to
fall short of the real allocator.

## What the estimate counts

Per card:

- the weights placed there, from the tensor table, following
  `--n-gpu-layers`, the CPU offload flags `--cpu-moe`, `--n-cpu-moe` and
  `--n-cpu-ffn`, and `--override-tensor`, then split across cards by
  `--tensor-split` or free memory;
- the KV cache of every layer that holds one: the header's full-attention
  interval or a per-layer KV head-count array picks out which layers do (a
  header naming neither charges every layer, and one with state sizes and
  no attention heads charges none); each entry is sized from the cache
  type settings, that layer's own KV head count where the header gives one
  per layer, and the header's key and value lengths (embedding size over
  head count when absent); all of it moves to RAM under
  `--no-kv-offload`;
- the recurrent state of every layer that holds no KV cache on a header
  carrying recurrent-state sizes, one copy per request slot, charged to
  the card that holds the layer (or to RAM where its KV would go); a
  header with state sizes and no attention heads is purely recurrent, and
  a layer with no KV heads but a non-zero per-layer feed-forward width is
  MLP-only and holds neither cache nor state;
- a compute buffer (see the formula below), counted only on a card that
  holds weights or KV, except under `split-mode row`, where it is counted
  once, on `--main-gpu`, and only when `--main-gpu` itself holds weights
  or KV; the logits term of that formula lands only on the one card that
  holds the output tensor;
- a fixed backend overhead, applied to every visible card whether or not
  it holds any weights or KV.

In RAM:

- the weights of layers kept off every card (block layers under a CPU
  rule or override, token embeddings, and the output matrix when it is
  kept off a card or tied to the embeddings);
- their KV cache and recurrent state;
- the context checkpoints: `--ctx-checkpoints` times the recurrent state
  of every recurrent layer, wherever that layer sits, since llama-server
  keeps its checkpoints as host vectors;
- a host compute buffer: the host term of the compute formula times the
  same per-card formula without its logits or vocabulary term (the host
  reserves a card-sized buffer for the same graph rather than one that
  follows the embedding size alone), plus the logits term itself when the
  output tensor stays in RAM instead of landing on a card;
- an output buffer the server hands back per slot: vocabulary size times
  four bytes times the slot count (the `--parallel` setting, else the
  engine's default of four on llama.cpp and one on ik_llama.cpp).

The compute buffer is a formula of f32 terms, each scaled by `--ubatch-size`
(itself capped to `--batch-size`, which is capped to the context): FFN
activations (active experts on a MoE model), the residual stream, recurrent
activations sized by the header's inner size, attention scores over the
context (dropped when flash attention is on or auto), a vocabulary-sized
activation charged to card compute buffers only, and logits sized by the
vocabulary, which is added only to the one card that holds the output
tensor, or to the RAM host buffer when that tensor stays there. The
vocabulary term is fitted along with the rest of `COMPUTE_TERMS`, still
charged to card compute buffers only; the host buffer excludes it, as it
excludes the logits term. `COMPUTE_TERMS` and `CARD_OVERHEAD_BYTES` hold the
coefficients and the fixed backend constant; `ENGINE_COMPUTE_SCALE`
multiplies every compute and host figure by one constant per engine, since
ik_llama.cpp's graphs reserve less than mainline's for the same model, and
an engine the table does not name keeps mainline's figures.

A draft model is placed by the same function, gated on the draft twins of
the offload flags (`spec-draft-override-tensor`, `spec-draft-n-cpu-moe`
and `spec-draft-cpu-moe`, which also accept upstream's
`--override-tensor-draft`, `--n-cpu-moe-draft` and `--cpu-moe-draft`
spellings), and its KV cache uses `--ctx-size-draft` when set and the main
context otherwise. Its weights, KV, recurrent state, checkpoints and
compute buffer add into the same per-card and RAM totals as the main
model's. A projector file's bytes add to `--main-gpu`'s weights unless
`--no-mmproj-offload` is set, in which case they go to RAM instead.

A sliding-window model whose header carries a per-layer window pattern
(`attention.sliding_window_pattern`, as Gemma 4 writes it) has each window
layer priced at the header's window head sizes (`key_length_swa`,
`value_length_swa`). On mainline llama.cpp without `--swa-full` a window
layer holds, per request slot, the smaller of the slot's share of the
context and the window plus one micro-batch, rounded up to a multiple of
256 tokens (with `--kv-unified`, one stream holding the window times the
slot count plus a micro-batch, capped at the context); with `--swa-full`
it holds the full context. The last `shared_kv_layers` layers hold no cache
of their own. A header with a sliding window but no pattern array is priced
at full context on every layer and labelled an upper bound wherever it
appears: the readout lines say "KV up to" instead of "KV", the tooltip
carries a separate upper-bound note, and the JSON carries
`kv_upper_bound: true`; the same label applies to any sliding-window model
on ik_llama.cpp, whose window cache is not modelled. `--swa-full` drops the
label in both cases, since the server then keeps the full window.

## The balanced tensor-split suggestion

Whenever the estimate is knowable, at least two cards are visible, and the
split mode places layers in contiguous runs (every mode but `row` and
`none`), the launcher searches for the `--tensor-split` value with the most
even per-card capacity and offers it as a suggestion. It is a suggestion
and never an assumption: with `--tensor-split` unset the estimate still
models the engines' own free-VRAM proportion, and the suggestion only ever
appears alongside that estimate, never in place of it.

A candidate is one of the whole-layer boundaries that leave no visible card
empty. It scores first on whether every card fits at the profile's context.
Among the candidates that fit, the next term is the minimum per-card
capacity, a card's margin (its free VRAM less its estimated total) divided
by its marginal cost per token, taking the minimum over the cards whose
marginal cost is non-zero, and then the minimum margin. A candidate that
leaves a card over budget is ranked on its minimum margin alone: a negative
margin divided by a cost is not a capacity, and treating it as one would
prefer the split with the deeper shortfall. A card's marginal cost per
token is measured at that candidate's
own split: its KV and state priced at the profile's context plus 4096
tokens, less the same figure at the profile's own context, divided by the
step, so window caps, cache types and slot rules need no second formula.

The search starts at the split the profile itself would use, and each
round takes the best single-layer move across any boundary while one
scores strictly better than the current candidate, stopping after at most
32 moves. On two cards with a single-peaked capacity curve this reaches
the best candidate; on three or more cards, on a curve with more than one
peak, or where the best candidate lies further than the move budget, it
can stop at a lower peak instead of the best one.

The value is rendered as layer counts per card, for example `38,28`: a
boundary given that way cannot round onto the wrong layer the way a
fractional `--tensor-split` value can. `--estimate --json` carries it
under `balanced_split` with the keys `value`, `layers_per_card`,
`boundary_layers` and `fits` (whether the split alone fits with no
offload); the key is absent whenever no split is computed, which covers
the `row` and `none` split modes, fewer than two visible cards, fewer
placed layers than cards, and an unknowable estimate.

A card shortfall message names the balanced split only where applying it
would be an improvement: it must place layers differently from the split
the profile already resolves to, and it must either fit on its own or need
a smaller offload count than the profile's own split does. A suggestion
that reproduces the placement the engine already performs is not offered,
since following it would cost the user `--fit` and change nothing else.
Where the split is named, the message says whether it fits on its own,
names the profile's current `--tensor-split` where one is set and says the
suggestion replaces it, and, where `--fit` would otherwise act, adds that
setting `--tensor-split` keeps it from acting. When the balanced split does
not fit on its own, the offload count search described above runs at both
splits, and text naming a count found at the balanced split says which
split it was found at.

## What the messages mean

A card shortfall names the card and how far its estimate exceeds that
card's free memory. Alongside it, the launcher searches for the smallest
`--n-cpu-moe` value on a MoE model (detected from the expert count or any
`_exps` tensor name) or `--n-cpu-ffn` value on a dense one at which every
card fits; on an engine that has no `--n-cpu-ffn` it offers the equivalent
`--override-tensor` pattern, an alternation of layer indices with
upstream's dense FFN regex. When offloading every layer still leaves a
card over budget, the message says no count fits and suggests lowering
the context or the KV cache type instead.

On llama.cpp, when `--fit` is unset or `on` and none of `--n-gpu-layers`,
`--override-tensor`, `--cpu-moe`, `--n-cpu-moe`, `--n-cpu-ffn` or
`--tensor-split` is set and the split mode is `layer` (or there is one
card), a shortfall also gets a note about what `--fit` will do. With no
`--ctx-size` set, the note names the context `--fit` will shrink toward
before it starts dropping whole layers, solved against the free VRAM
summed across cards less each card's `--fit-target` margin (1024 MiB by
default) and never below the `--fit-ctx` floor (4096 by default). With
`--ctx-size` set, `--fit` keeps that context instead and moves whole
layers to RAM (experts first on a MoE model), and the note says so
without naming a predicted context. That note reaches the abortable
launch dialog only when `--fit` is left unset; an explicit `on` keeps the
launch silent and the note stays in the readout and tooltip only.

On ik_llama.cpp, when `--fit` is explicitly `on`, a MoE model that does
not fit gets a note that ik keeps the experts of as many layers as needed
in RAM instead, with the context and layer count unchanged and the RAM
line growing by the same amount; a dense model gets a warning that ik
refuses to load it, which does reach the dialog since that failure is
fatal. With `--fit` unset or `off` on ik, only the plain shortfall
applies.

A RAM warning fires when the RAM total exceeds the launch node's
available memory. It is always a warning, never a refusal by itself: its
wording says the launch will fail when `--load-mode` locks every page
(`none`, `mlock` or `mmap+mlock`, or `--mlock` set) and otherwise that the
server will page weights in and out of RAM and run slowly.

## Calibrating `COMPUTE_TERMS`, `ENGINE_COMPUTE_SCALE` and `CARD_OVERHEAD_BYTES`

All three live in `src/llama_launcher/core/vram.py` and are fitted against
llama.cpp's and ik_llama.cpp's own logged numbers, not derived from either
engine's source. `COMPUTE_TERMS` and `ENGINE_COMPUTE_SCALE` are f32
coefficients per micro-batch token and engine multipliers respectively;
`CARD_OVERHEAD_BYTES` is a byte count. SPEC 2.19 sets the tolerance a fit
must meet: KV plus recurrent state reads high by at most a quarter and
never low on the sum across cards (on the RAM figure instead, for a record
with no card figures), and per card within one layer's KV of that;
compute and host buffers never read low and read at most two and a
half times high, compared per card as sorted figures since the output card
follows settings the records do not carry. `COMPUTE_TERMS` and
`ENGINE_COMPUTE_SCALE` are fit against six calibration records measured
2026-09-06. A sweep in the
Benchmark tab gives one combined measured against estimated total per card
and for RAM, a quick check on the estimate as a whole; refitting the three
constants still needs the manual procedure below, because the compute
figure has to be read on its own and the overhead comes from the exit-time
memory breakdown table, which a sweep's log read never sees since it
happens while the server is still running.

Each engine logs its buffers differently, so the read step splits by
engine; steps 1, 4, 5 and 6 are the same for both.

### llama.cpp (mainline)

Set the profile's Verbosity setting (the Logging group, `-lv`) to 4:
llama.cpp 0.4.0 prints the per-device buffer lines and the
`common_memory_breakdown_print` table (`llama_memory_breakdown_print`
before 0.4.0) only at verbosity 4 and above, nothing at its default of 3.
The table appears once before loading and once just before the server
exits; every line carries a timestamp and level prefix such as
`0.05.529.870 I`. Read the last occurrence. It has one row per device,
each row reading `total = free + self + unaccounted` with
`self = model + context + compute`.

### ik_llama.cpp

ik_llama.cpp prints its buffer lines at its default verbosity already, in
the `llm_load_tensors: CUDA0 buffer size` form, with no separate
memory-breakdown table; setting verbosity 4 there only adds a line per
token and changes nothing about the buffer lines themselves.

### Procedure

1. Before launching, read each card's already-used VRAM with `nvidia-smi
   --query-gpu=index,memory.used --format=csv`, so the per-card subtraction
   in step 5 pairs unambiguously on a multi-card box.
2. Launch a dense model profile and a MoE model profile per engine
   headlessly with
   `llama-launcher --launch --profile NAME` (detached, kept after exit) or
   with **Run detached** enabled in the GUI. A foreground GUI launch runs
   its container with `--rm`, so the container and its log are gone at
   exit; the buffer lines still stream into the Monitor tab's log, so read
   it there instead when launching in the foreground.
3. Stop each server with `llama-launcher --stop --profile NAME` (or the
   GUI's Stop), then read `podman logs <container name>` (`docker logs`
   likewise); find the container name with `podman ps -a`, or read it off
   the Monitor card. Read the buffer lines the engine's section above
   names.
4. Run `llama-launcher --estimate --profile NAME --json` for the same
   profile.
5. Compare the JSON's `estimate.cards[i].compute` against the logged
   compute figure, and `estimate.cards[i].overhead` against the logged
   unaccounted figure minus the VRAM that step 1 found already used on
   that card: the launcher's own estimate already judges each card against
   its free VRAM at probe time, so other processes' usage has to come out
   of unaccounted before the two overheads are comparable.
6. Add the measurement as a new record in `tests/core/calibration_records.py`
   (header block, settings, free VRAM per card and the logged buffer
   sizes, all in bytes), then run `scripts/fit_compute_terms.py`: it
   refits `COMPUTE_TERMS` and `ENGINE_COMPUTE_SCALE` over every record at
   once and prints the winning point, or the least-bad one when SPEC
   2.19's tolerance is not reachable on every record together.
7. Adjust `CARD_OVERHEAD_BYTES` by hand from the unaccounted comparison in
   step 5, since the fit script does not touch it; stop once every card's
   overhead sits within a few hundred MiB of the logged figure and never
   reads lower than it.

## JSON keys

`memory_fit.to_json`, surfaced under the `estimate` key of `--estimate
--json`, carries:

- `fits`, `ctx`, `kv_upper_bound`;
- `cards`: a list, each with `index`, `est`, `free`, `margin`, `fits`,
  `weights`, `kv`, `compute`, `overhead`, `state`;
- `ram`: `est`, `available`, `margin`, `fits`, `weights`, `kv`, `buffers`,
  `state`, `checkpoints`;
- `messages`: a list of the message strings shown in the readout;
- `balanced_split`, only when a split is computed: `value`,
  `layers_per_card`, `boundary_layers`, `fits`.

## Known limits

- The compute buffer is a formula (FFN, residual, recurrent activations,
  attention scores without flash attention and logits on the output card,
  scaled by `--ubatch-size`, plus the fixed per-card overhead and, per
  engine, the `ENGINE_COMPUTE_SCALE` multiplier), not a readout of the
  engine's own allocator; every place it is shown marks it approximate.
- A sliding-window model without a pattern array, or on ik_llama.cpp, is
  the labelled upper bound above.
- The RAM estimate does not model `CPU_REPACK`, the second, repacked copy
  of weights llama.cpp keeps on a CPU-only launch (or of expert layers the
  offload knobs keep in RAM) beside the mapped one; the estimate reads low
  by that amount there.
- llama.cpp's real `--fit` distributes layers per card while it searches;
  the predicted context here is solved against the free VRAM summed
  across every card, not per card.
- The file-size fallback triggers when a GGUF's tensor table cannot be
  parsed from the 64 MiB header read; the fallback bytes are then spread
  across cards by `--tensor-split` or free-memory proportion, the same way
  a real per-layer table would land (a single visible card gets all of
  them, and so does `--main-gpu` under `--split-mode none`).
- A model file the launcher cannot reach through a local mount (a path
  under no mount, or a file that lives only on a remote node) yields no
  estimate at all: the readout stays empty and `--estimate` exits 2.
- The Configure readout's cache stamp covers every part named by the
  model's part-of-total filename suffix (modification time and size for
  each), so a changed sibling part is noticed on the next render; a part
  that lives outside that naming convention is still missed.
- `--device` is not modelled: every visible card is counted and gets the
  per-card overhead even when the launch excludes it.
- The balanced split moves whole layers only; moving tensor families
  across the boundary with `--override-tensor` is not modelled, out of
  scope by SPEC section 4.
