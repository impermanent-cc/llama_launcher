# llama_launcher: roadmap

Direction beyond the current cycle. Items move to SPEC.md when grilled and
settled, then to TASKS.md when a cycle picks them up.

## Next

- Model a tied output tensor that stays in host RAM. All three Gemma 4 runs
  of 2026-09-06 kept the tied token embedding on the host with every layer
  offloaded, the logged host model buffer being exactly vocabulary times
  embedding size at the file's quantization, while the estimate puts the
  output tensor on a card whenever every layer is offloaded. The compute
  terms absorb the difference within tolerance today, so this is accuracy
  rather than a failing gate: the logits term is charged to a card that the
  run kept empty, and the host compute buffer is correspondingly light.
- Carry the balanced split into the offload sweep: each sweep point is
  measured at whatever split the profile carried, so Apply writes a count
  alone. Decide whether the sweep should vary the split as well as the
  count, which is the placement sweep below, or whether Apply should write
  the balanced split for its winning count.
- The per-card overhead and the compute buffers are constant while only the
  weights and KV follow --tensor-split, so on the 5080 plus A2000 box a
  proportional auto split over-burdens the second card. The balanced split
  suggestion answers this for a profile that asks for it; whether the
  estimate should also model a main card that carries as much as fits is
  still open.
- Review the remaining flags both engines accept that the catalog exposes
  for neither: the control-vector family (repeatable, one takes two tokens,
  needs panel plumbing like LoRA) and --spec-replace (two tokens). Decide
  per flag: add with plumbing, or record as out of scope.
- Periodic re-audit of both engines: common/arg.cpp against
  settings_catalog, root and ggml CMakeLists.txt against build_catalog.
- Documentation cycle: README and CHANGELOG to ASCII and dash-free so they
  leave the guard allowlist (RPC.md left it on 2026-09-10); reword the
  legacy ` -- ` comment separators and turn the doubled-hyphen guard on.

## Later

- After the offload sweep: a ubatch sweep on the same runner, a headless
  --sweep command, and sweeps on a remote node through the node's podman.
- Placement sweep on the same runner: vary --override-tensor moves of whole
  blocks or tensor families across the --tensor-split boundary on a two-card
  box, score by the fitted context the server logs (n_ctx_slot) and by the
  graph split count, and rank by per-card capacity (free VRAM over that
  card's per-token KV cost, minimum across cards); the same rule replaces
  the summed-budget predicted context. Log lines to add to the parser: RS
  buffer size, graph splits, n_ctx_slot.
- Reproduce mainline fit's per-card layer distribution for the predicted
  context after shrinking.
- Re-audit ik_llama.cpp's fit and split mode graph when they change.
- Live multi-node testing on a GPU worker; pooled inference across
  CPU-only rpc-servers crashes upstream and cannot be tested here.
- Server-mode --api-key delivered through a key file (mount plumbing; low
  severity on a single-user desktop).
- A model-file-existence warning before launch, routed so that it does not
  fire through the router's own health path.
- A warning when the profile's GPU mode is set but the launch node has no
  GPU; it needs a probe that does not run on every poll tick.
- Render an nvidia-smi "[N/A]" field as unknown instead of failing the
  parse; GpuStat's integer fields would have to admit None.
- Settle pool_preflight's double count on a worker that both pledges memory
  and is probed, during live node testing.
- Benchmark with the user's own prompt or prompts, so the run reports what
  the model did with them rather than a bare throughput figure.
- Queue models or profiles to run one after another, with a way to show
  each finished run's answers, for example by opening the web UI on the
  finished chats.
- The Speculative Decoding group is the widest in the settings column once
  the tools row is two columns, from bool rows whose checkbox text repeats
  the flag beside the row label; shorten or wrap them.

## Not planned

- HF download flags and URL models: they bypass the local-path model and
  the GGUF preflight.
- CodeQL: costs Actions minutes for little gain here.
- Reranker GGUF auto-detection: a bge reranker reports arch bert with no
  metadata signal, and filename heuristics are ruled out.
- Stripping host detail from the diagnostic report: it exists for bug
  reports and the detail is the point.
