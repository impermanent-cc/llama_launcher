# VRAM and RAM: how the memory estimate is built

Llama Launcher's memory estimate reads a model's GGUF tensor table and
places every tensor the way llama.cpp and ik_llama.cpp place it, so the
Configure readout, the launch preflight dialog and `llama-launcher
--estimate` all show the same numbers for the same profile. This document
describes what the estimate counts, what its messages mean, how to
recalibrate its two hand-tuned constants, its JSON keys, and where it is
known to fall short of the real allocator.

## What the estimate counts

Per card:

- the weights placed there, from the tensor table, following
  `--n-gpu-layers`, the CPU offload flags `--cpu-moe`, `--n-cpu-moe` and
  `--n-cpu-ffn`, and `--override-tensor`, then split across cards by
  `--tensor-split` or free memory;
- the KV cache of every layer whose card that is (all of it moves to RAM
  under `--no-kv-offload`);
- a compute buffer (see the formula below), counted only on a card that
  holds weights or KV, except under `split-mode row`, where it is counted
  once, on `--main-gpu`, and only when `--main-gpu` itself holds weights
  or KV;
- a fixed backend overhead, applied to every visible card whether or not
  it holds any weights or KV.

In RAM:

- the weights of layers kept off every card (block layers under a CPU
  rule or override, token embeddings, and the output matrix when it is
  kept off a card or tied to the embeddings);
- their KV cache;
- a host output buffer sized from the vocabulary and the logical batch
  size (`--batch-size`, capped to the context; `--ubatch-size` is
  separately capped to that batch).

A draft model is placed by the same function, gated on the draft twins of
the offload flags (`spec-draft-override-tensor`, `spec-draft-n-cpu-moe`
and `spec-draft-cpu-moe`, which also accept upstream's
`--override-tensor-draft`, `--n-cpu-moe-draft` and `--cpu-moe-draft`
spellings), and its KV cache uses `--ctx-size-draft` when set and the main
context otherwise. Its weights, KV and compute buffer add into the same
per-card and RAM totals as the main model's. A projector file's bytes add
to `--main-gpu`'s weights unless `--no-mmproj-offload` is set, in which
case they go to RAM instead.

A sliding-window model's KV estimate uses the full context for every
layer rather than the model's own window, so it is labelled an upper
bound wherever it appears: the readout lines say "KV up to" instead of
"KV", the tooltip carries a separate upper-bound note, and the JSON
carries `kv_upper_bound: true`. Setting `--swa-full` drops the label,
since the server then really does keep the full window.

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

## Calibrating `COMPUTE_TERMS` and `CARD_OVERHEAD_BYTES`

Both constants live in `src/llama_launcher/core/vram.py` and are
hand-tuned against llama.cpp's own numbers, not derived from its source.
`COMPUTE_TERMS` are f32 elements per micro-batch token; `CARD_OVERHEAD_BYTES`
is a byte count. A sweep in the Benchmark tab gives one combined measured
against estimated total per card and for RAM, a quick check on the
estimate as a whole; refitting `COMPUTE_TERMS` and `CARD_OVERHEAD_BYTES`
still needs the manual procedure below, because the compute figure has to
be read on its own and the overhead comes from the exit-time
`llama_memory_breakdown_print` table, which a sweep's log read never sees
since it happens while the server is still running. To refit them:

1. Before launching, read each card's already-used VRAM with `nvidia-smi
   --query-gpu=index,memory.used --format=csv`, so the per-card subtraction
   in step 5 pairs unambiguously on a multi-card box.
2. Set the profile's Verbosity setting (the Logging group, `-lv`) to 4:
   llama.cpp 0.4.0 prints the per-device buffer lines and the memory
   breakdown table only at verbosity 4 and above, and nothing at its
   default of 3. Then launch a dense model profile and a MoE model profile
   headlessly with
   `llama-launcher --launch --profile NAME` (detached, kept after exit) or
   with **Run detached** enabled in the GUI. A foreground GUI launch runs
   its container with `--rm`, so the container and its log are gone at
   exit; that same table still streams into the Monitor tab's log, so read
   it there instead when launching in the foreground.
3. Stop each server with `llama-launcher --stop --profile NAME` (or the
   GUI's Stop), then read `podman logs <container name>` (`docker logs`
   likewise); find the container name with `podman ps -a`, or read it off
   the Monitor card. llama.cpp prints the `common_memory_breakdown_print`
   table (`llama_memory_breakdown_print` before 0.4.0) once before loading
   and once just before it exits; read the last one. Every line carries a
   timestamp and level prefix such as `0.05.529.870 I`. The table has one
   row per device, each row reading `total = free + self + unaccounted`
   with `self = model + context + compute`.
4. Run `llama-launcher --estimate --profile NAME --json` for the same
   profile.
5. Compare the JSON's `estimate.cards[i].compute` against the table's
   compute column, and `estimate.cards[i].overhead` against the table's
   unaccounted column minus the VRAM that step 1 found already used on
   that card: the launcher's own estimate already judges each card against
   its free VRAM at probe time, so other processes' usage has to come out
   of unaccounted before the two overheads are comparable.
6. Adjust `COMPUTE_TERMS` or `CARD_OVERHEAD_BYTES`, re-run the estimate,
   and repeat; re-read the log (step 3) before relaunching the profile,
   since relaunching removes the previous stopped container.
7. Stop once every card's compute and overhead sit within a few hundred
   MiB of the logged numbers and never read lower than them: an estimate
   that reads a little high is a safe warning, one that reads low hides a
   real shortfall.

## JSON keys

`memory_fit.to_json`, surfaced under the `estimate` key of `--estimate
--json`, carries:

- `fits`, `ctx`, `kv_upper_bound`;
- `cards`: a list, each with `index`, `est`, `free`, `margin`, `fits`,
  `weights`, `kv`, `compute`, `overhead`;
- `ram`: `est`, `available`, `margin`, `fits`, `weights`, `kv`, `buffers`;
- `messages`: a list of the message strings shown in the readout.

## Known limits

- The compute buffer is a formula (logits, FFN, residual and, without
  flash attention, attention-score terms scaled by `--ubatch-size`, plus
  the fixed per-card overhead), not a readout of llama.cpp's own
  allocator; every place it is shown marks it approximate.
- A sliding-window model's KV figure is the labelled upper bound above;
  its per-layer window pattern is not read.
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
