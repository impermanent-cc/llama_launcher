# llama_launcher: tasks

## Current phase

Idle: no cycle open. The estimate calibration cycle (fix/estimate-calibration,
SPEC.md 2.18, 2.19, 2.26, 2.29, 2.31 and 2.32) landed on main on 2026-09-06.
The owner smokes that need the 5080 plus A2000 box are listed below: the
re-run of --estimate against the sweep files first, then a dense non-hybrid
measurement to pin the FFN and residual terms, then the two GUI checks the
2026-09-06 runs left uncovered (a failed sweep point plus Apply, and the
launch dialog). Repository chores the owner does by hand close the open
items list.

## Open items

- [ ] validation._is_active re-derives command_builder's emit rule rather
      than calling it, and does not model the load-mode suppression of
      no-mmap and mlock or the engine_value SKIP that drops an enum left at
      its default. Its docstring says so. The webui pair is the SKIP case and
      is masked today only by the engine mismatch.
- [ ] _is_active reads an int-typed setting carrying the string "0" as
      active, which contradicts its own "a zero count does nothing".
      Hand-edited profile JSON only.
- [ ] capabilities._sug_ctx is the last context reader still on raw
      ctx-size, so a profile using --kv-unified-per-slot past the model's
      trained context gets no context suggestion.
- [ ] _rel_moe now decides a dense-only tier too, so its name no longer
      covers what it returns; and an embedding model, being dense, shows a
      RECOMMENDED dot on n-cpu-ffn. Spec-conformant, not useful.
- [ ] configure_panel calls self.current_profile() more than once per
      debounced render (once in _render_fit_line and again inside
      _current_fit_report).
- [ ] fit_report runs on the UI thread per debounced edit and costs about
      0.7 s on a model with tens of thousands of tensors (the suggestion
      bisection re-places every tensor up to eight times); cache the
      placement per tensor table and settings, or move the render off
      thread.
- [ ] The launch click probes the GPUs and RAM synchronously on the UI
      thread (two ssh round trips on a remote node, up to ten seconds
      frozen); the Configure panel's off-thread gather with its TTL cache
      could serve the preflight instead.
- [ ] --device is not modelled: every visible card is counted and gets the
      per-card overhead even when the launch excludes it (VRAM.md, known
      limits).
- [ ] The suggested --override-tensor value in a shortfall message is
      unquoted and, if pasted, replaces rather than extends an existing
      override; a shortfall under 100 MiB renders as "~0.0 GiB".
- [ ] A draft model or projector under no mount counts as zero bytes in the
      estimate with no note; --estimate exits 0 when only RAM is over
      budget.
- [ ] ik_llama.cpp layer mode fills cards by cumulative bytes rather than
      layer index, so per-card weights drift on uneven (cpu-moe) layers;
      placement.distribute's engine parameter is unused.
- [ ] Router readout sums each member's gpu_total, so every member adds the
      per-card overhead and its compute buffer; visibly inflated for four or
      more members.
- [ ] Split-model parts hardlinked under two names with a disagreeing
      split.count key count twice (malformed layout only); the Configure
      cache stamps only the first part.
- [ ] tests/core/test_purity.py's enhancement-module test is a strict
      subset of the whole-core scan.
- [ ] Three LaunchController calls into the panel's private
      _cached_meta_weights; promote it to a public method.
- [ ] No test asserts that an engine-gated build_catalog option carries a
      tooltip.
- [ ] video-timestamp-interval takes a minimum of 0 and nothing here
      establishes what upstream does with 0.
- [ ] src/llama_launcher/ui/widgets/setting_widgets.py:147 carries a
      doubled-hyphen prose separator and narrates history in a comment; one
      for the documentation cycle's prose sweep, along with RPC.md's
      non-ASCII at lines 143 and 163.
- [ ] core.sweep.parse_load_log drops an "output buffer" line whose device
      is a card, and knows no "RS buffer size" kind, so a hybrid or SSM
      model's measured column reads short (RS is on the ROADMAP with the
      placement sweep).
- [ ] The Benchmark panel's reset() leaves the previous profile's sweep
      table on screen, and sweep_store.load has no production caller: a
      profile's stored sweep is never shown again after a restart.
- [ ] services.sweep._wait_ready duplicates headless.wait_ready minus the
      cancel flag and never probes at timeout 0; _read_log and
      _stop_and_remove have no test through a fake subprocess.run, and the
      QThread and cancel paths are covered by the owner smoke only.
- [ ] No refresh_sweep after a sweep ends, so a fit render during a run can
      leave "A benchmark or sweep is already running." on the status line;
      the prompt-sizes parse is duplicated between panel and controller.
- [ ] tests: sweep_store tests lock neither the 0600 mode, the sweeps/
      parent nor that a second save replaces the first;
      test_sweep_counts_inclusive_clamped_and_deduped names a dedupe that
      does not exist; the sweep controller tests patch away
      smallest_fitting_offload so the kwargs plumbing is unexercised.
- [ ] The sweep prefill is memoized on its computed values, so a typed
      range survives a switch to a profile with the identical computed
      range, and a start-path refusal ("same port") stays on the status
      line until the availability reason itself changes.
- [ ] The RAM estimate does not model llama.cpp's CPU_REPACK buffer: on a
      CPU-only launch the server keeps a repacked second copy of the
      weights it runs on the CPU (1.2 GiB beside the 2.5 GiB mmap of the
      e2b model on 2026-09-06, measured 3.8 GiB against 3.4 estimated).
      The same applies to expert layers kept in RAM by the offload knobs.
- [ ] The host multiplier in COMPUTE_TERMS also scales the attention-scores
      term, which no calibration record covers since every record ran with
      flash attention on.
- [ ] scripts/fit_compute_terms.py's search() computes the best_effort point
      on every grid combination even after a feasible one is found, so the
      full grid runs regardless of how early the search succeeds.
- [ ] The checkpoints term charges the --ctx-checkpoints maximum (32 by
      default) times the recurrent state per slot to RAM, although the
      server creates checkpoints on demand; no log line measures the term.
      Watch a long session on the 35B-A3B for RAM growth into it.
- [ ] Multi-head latent attention (deepseek2 and kin) is priced at the
      header's per-head key and value lengths, far above the engine's
      latent cache; reading kv_lora_rank plus the rope dimension would fix
      it (SPEC out of scope).
- [ ] With no tensor table, an --override-tensor rule that promotes the
      output tensor to a card while -ngl leaves the output layer in RAM
      still sends the logits term to the host buffer; the device flag wins
      over the override.
- [ ] recurrent_layer_mask treats any layer with a non-zero per-layer
      feed-forward width as not recurrent, so a hybrid whose recurrent block
      also carries an MLP would be charged no state (SPEC 2.32 as written).
- [ ] The ik calibration record's card 0 KV lower bound clears by a few KiB
      only through the log-precision allowance; a second ik measurement
      would settle whether the one-layer slack is enough.
- [ ] Owner, GitHub: ask GitHub Support to garbage-collect the unreachable
      objects left by the 2026-08-31 history rewrite. The unredacted
      router.png blob (91a946f) was still fetchable by exact SHA on
      2026-09-02 and the repository is public; nothing records the request
      as filed.
- [ ] Owner, GitHub: the repository has no topics, and README carries no CI
      badge although Actions has been green since 2026-09-01.

## Pending owner smokes

- [ ] Re-run `--estimate --json` for the 27B dense and 35B-A3B profiles on
      the 5080 plus A2000 box and compare each card's `kv + state + compute`
      and the RAM `buffers` against the sweep files' measured figures; then
      re-run one sweep per profile and report the measured against
      estimated columns.
- [ ] Measure one dense non-hybrid model on the 5080 plus A2000 box (any
      Llama or Gemma dense GGUF, mainline, Verbosity 4, one sweep point or
      a detached launch with the buffer lines and the exit table) and paste
      the print_info block plus the buffer lines, so a fifth record can pin
      the FFN and residual terms that the hybrid records leave free.
- [ ] On the 27B dense profile, run a sweep whose "from" sits a few counts
      below the prefilled start so the first point is too small: confirm it
      records as failed with its log line in the tooltip and the sweep
      continues; then confirm the winner is marked and Apply writes the
      count into the profile. The prefilled-range run itself is done (below).
- [ ] Open the launch dialog on KDE/Wayland with an over-budget context and
      confirm it shows the per-card breakdown, the --fit note and the
      suggested offload count. The readout and its tooltip are confirmed
      (below); the dialog is not.
- [x] Run a sweep on the 5080 plus A2000 box on the 27B dense profile with
      the prefilled range. Done 2026-09-06, files in
      DevDocs/llama_launcher/calibration-2026-09-06: n-cpu-ffn 0 to 8 on the
      27B dense (ctx 32768, tensor-split 60,40, q8_0 KV) and n-cpu-moe 2 to
      10 on the 35B-A3B MoE, five points each, every point launched,
      benchmarked, stopped and recorded ok in 5 to 6 s to ready, with the
      measured model, KV and compute columns filled. The measured columns
      were not within a few hundred MiB: KV read six times high on the dense
      and twice high on the MoE, which became the estimate calibration
      cycle.
- [x] Calibrate the compute constants on the 5080 plus A2000 box. Done
      2026-09-06 from the two sweep files above, the ik_llama.cpp launch log
      at Verbosity 4 and a CPU-only e2b run on the dev box: the four records
      live in tests/core/calibration_records.py and
      scripts/fit_compute_terms.py refit COMPUTE_TERMS from them
      (fix/estimate-calibration). CARD_OVERHEAD_BYTES is still unmeasured;
      it needs the exit-time table.
- [x] Launch a real profile and confirm the four-line readout and its
      tooltip on KDE/Wayland. Done 2026-09-06 during the ik smoke below: one
      line per card and one for RAM, the tooltip carrying the --fit note and
      the offload suggestion. The suggested counts (--n-cpu-ffn 3 and
      --n-cpu-moe 24) came from the pre-calibration estimate and were too
      high; the launch dialog is the open item above.
- [x] Smoke an ik_llama.cpp profile with --fit on. Done 2026-09-06 on the
      5080 plus A2000 box with Qwen3.6-35B-A3B MXFP4 on the cu13-server
      image: the command carried bare --fit, the server ran with it, and
      the readout tooltip carried the ik note (24 layers' experts in host
      RAM). The note was wrong in substance: ik loaded all 41 layers on the
      cards with 1.2 and 1.8 GiB to spare, the KV over-estimate again.
- [x] Smoke the new flags against a real llama.cpp 0.4.0 image. Done
      2026-09-05 against ghcr.io/ggml-org/llama.cpp:server-b10818 (version
      0.4.0-dev, build 10818), the first server tag past b10795; the
      floating :server tag was still b10795 that morning. A profile carrying
      --parallel 2, --kv-unified-per-slot 2048, --no-reasoning-preserve,
      --video-fps 2.0, --video-timestamp-interval 1000 and
      --video-ffmpeg-dir /usr/bin plus the e2b mmproj: --dry-run emitted all
      of them with no validation issue and exit 0; --launch --wait reached
      "ready", /health returned ok, /props reported build b10818 with two
      slots of 2048, and the server log carried no unknown-argument line.
      --stop exited 0.
- [x] Launch a real profile and confirm the UI text renders unchanged.
      Done 2026-09-03 on KDE/Wayland through XWayland (QT_QPA_PLATFORM=xcb):
      the e2b profile loaded through the profile combo's activated signal,
      the command preview matched `--dry-run --profile e2b` exactly, and
      the glyphs rendered as glyphs, not tofu. An AST comparison of every
      non-ASCII string constant in src/ against the pre-cycle tree found 66
      distinct constants over 81 occurrences, all byte-identical.
- [x] Open a fresh session in this repo and confirm the session-start
      summary prints this file's current phase. Done 2026-09-03 by running
      ~/.claude/hooks/session-start.sh, which is what a session runs; it
      printed the branch, the last ten commits and this file's phase.
- [ ] Live multi-node test on a GPU worker when one is available. Still
      blocked: nvidia-smi is absent on this box and the Stats dock reports
      "GPU: unavailable".

## Done this cycle

- The VRAM estimate charges KV only to layers that hold a cache, from the
  header's full-attention interval or a per-layer head-count array, sizes
  entries from the header's key and value lengths and the cache types, and
  adds the recurrent state of Gated DeltaNet layers per request slot and,
  in RAM, the context checkpoints the server keeps for them. On the two
  Qwen3.5 and 3.6 hybrids measured on 2026-09-06 the KV sum now lands
  within 2.5 percent of the server's own figure; before, it read six times
  high on the 27B dense.
- The compute term charges the logits buffer to the card that holds the
  output tensor only, gains a recurrent-activations term, and is scaled per
  engine (ik_llama.cpp 0.7). RAM carries a host compute buffer as a fraction
  of the card formula and an output buffer of vocabulary times four bytes
  times request slots, in place of the 2 GiB batch-sized buffer of before.
- A calibration records module holds the four measured runs and a fit
  script refits the term table against them; every record reads never low,
  compute and host at most 2.45 times high, KV plus state within a quarter.
- The sweep parser reads ik_llama.cpp's buffer lines and recurrent-state
  lines; sweep launches raise log verbosity on mainline llama.cpp only; the
  sweep's estimated column includes the recurrent state.
- The GGUF reader exposes the interval, per-layer KV heads, head sizes and
  recurrent sizes; the readout, tooltip and JSON show state and
  checkpoints; VRAM.md documents the terms and a per-engine calibration
  procedure.
- The suite went from 1994 to 2039 tests.
