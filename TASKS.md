# llama_launcher: tasks

## Current phase

Idle: no cycle open. The last cycle, feat/tensor-split-rebalance, landed on
2026-09-07: the memory estimate computes a capacity-balanced
`--tensor-split` and offers it, in `--estimate --json` always and in a card
shortfall message, without changing anything about what the estimate
assumes. SPEC.md 2.33 and 2.34 carry the rules and section 4 carries the
new out-of-scope bullet. Next: the owner smoke below decides whether the
run becomes a seventh calibration record.

## Open items

- [ ] Where no offload count fits at the profile's own split but the named
      balanced split fits on its own, the shortfall message pairs "no
      offload count fits; lower the context or the KV cache type" with a
      split that already solves it. Both sentences are true, the advice is
      stale: seven such states in a 360 case sweep.
- [ ] The ik_llama.cpp fit-on MoE branch emits a card shortfall message
      that names no balanced split, although FitReport.balanced and the
      JSON carry one for the same profile.
- [ ] The balanced-split sentence names only the first boundary layer, so
      on three or more cards it describes one boundary of several; no test
      exercises fit_report with three cards.
- [ ] balance.marginal_bytes_per_token's step parameter is passed by no
      caller or test, and _at_least_one's len(out) > total guard is
      unreachable from its only caller.
- [ ] No test pins that draft_meta, draft_weights and mmproj_bytes reach
      the balanced search: dropping them from the fit_report call would
      break nothing.
- [ ] One debounced render runs the balanced search twice on a shortfall.
      configure_panel's _set_fit_line emits fit_rendered, main_window wires
      it to benchmark_controller.refresh_sweep, and its sweep_prefill calls
      panel._current_fit_report() again: two fit_report calls, two
      searches, twelve estimate_memory calls.
- [ ] balance.balanced_split reaches the highest-scoring candidate on two
      cards with a single-peaked capacity curve, and can stop at a lower
      peak on three or more cards, on a curve with more than one peak, or
      where the best candidate lies more than 32 moves from the start
      (SPEC 2.33 as written).
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
      thread. A shortfall now also runs the balanced search and two offload
      bisections: measured 0.077 s stubbed against 0.443 s on a 32k-tensor
      table.
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
- [ ] An MTP draft (--spec-type draft-mtp, gemma4-assistant) reserves a
      3320 MiB compute buffer on card 0 for 58 MiB of weights and shares the
      main model's KV (shared_kv_layers 4); the estimate has no term for it
      and charges the draft KV as if it had its own cache.
- [ ] With --n-gpu-layers all and no tensor table the estimate puts the
      output layer on a card, but all three Gemma 4 runs of 2026-09-06 kept
      the tied token embedding in host RAM (the logged host model buffer is
      exactly vocabulary times embedding size at the file's quantization),
      so the logits landed in the host compute buffer and the card buffers
      came out symmetric. The fitted compute terms absorb the difference
      within tolerance, so this costs accuracy rather than failing a gate.
- [ ] The host compute buffer's vocabulary block is f16 on the Gemma runs
      (262144 x 512 x 2 bytes = 256 MiB), while the compute formula prices
      every activation in f32; the fitted logits coefficient absorbs the
      factor rather than the formula naming the width.
- [ ] calibration_records.py declares output_card in its docstring but
      nothing in the repository reads it; either the compute assertion uses
      it or the key goes.
- [ ] test_calibration.layer_kv_bytes prices a card's KV slack at the
      full-attention head size, so on a sliding-window record the per-card
      band is about twice the largest real layer and only the summed 1.25
      check binds.
- [ ] The draft model's memory estimate uses the main model's micro-batch
      size and slot count, although llama.cpp builds the draft context with
      one sequence and its own batch; the pre-existing context fallback
      makes the same approximation.
- [ ] kv_layer_mask clears the shared_kv_layers tail before
      recurrent_layer_mask reads it, so a header carrying both recurrent
      state sizes and shared_kv_layers would charge recurrent state to its
      trailing shared layers. No such header exists today.
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

- [ ] Take a profile that is over budget on one card on the 5080 plus
      A2000 box, apply the suggested `--tensor-split` from
      `--estimate --json`, launch at Verbosity 4 and check the per-card
      model buffer lines against the predicted boundary layer and the KV
      figures against the predicted per-card marginal cost. The boundary
      layer must match exactly and each card's model buffer must land
      inside the tolerance the calibration records use. With complete
      buffer lines the run becomes a seventh entry in
      tests/core/calibration_records.py; without them it stays a reported
      smoke.
- [ ] Re-run `--estimate --json` for the 27B dense and 35B-A3B profiles on
      the 5080 plus A2000 box and compare each card's `kv + state + compute`
      and the RAM `buffers` against the sweep files' measured figures; then
      re-run one sweep per profile and report the measured against
      estimated columns.
- [ ] Re-read the g31b_ud4k container's log once the run has finished: the
      2026-09-06 paste ends at the model buffer lines, with no KV or
      compute figures.
- [x] Measure one dense non-hybrid model on the 5080 plus A2000 box. Done
      2026-09-06 with Gemma 4 12B (dense, MTP draft), 26B-A4B (MoE) and 31B
      (dense, model buffers only), mainline b10818 at Verbosity 4 with
      --swa-full; print_info blocks, buffer lines, commands and estimate
      JSON in DevDocs/llama_launcher/calibration-2026-09-06. Weights are
      exact on every card; KV reads 1.9x to 2.2x high and compute reads low
      on card 0, see the open items. Not yet added as records: the KV rule
      fails until the reader knows the sliding-window head size.
- [ ] On the 27B dense profile, run a sweep whose "from" sits a few counts
      below the prefilled start so the first point is too small: confirm it
      records as failed with its log line in the tooltip and the sweep
      continues; then confirm the winner is marked and Apply writes the
      count into the profile. The prefilled-range run itself is done (below).
- [x] Open the launch dialog on KDE/Wayland with an over-budget context and
      confirm it shows the per-card breakdown, the --fit note and the
      suggested offload count. Confirmed by the owner on 2026-09-06.
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

- The memory estimate computes a capacity-balanced `--tensor-split`: the
  candidates are whole-layer boundaries leaving no card empty, scored on
  feasibility, then the minimum per-card capacity (a card's margin over its
  marginal bytes per token, minimized over the cards whose cache grows with
  the context), then the minimum margin. The search starts at the split the
  profile itself would use and takes the best single-layer move while one
  scores strictly better, to a bound of 32 moves.
- A card's marginal cost per token is measured, not derived a second time:
  its KV and state priced one 4096-token step away, divided by the step, so
  window caps, `--swa-full`, unified against per-slot caches and quantized
  cache types follow the rules already calibrated.
- The split is a suggestion and never an assumption: with `--tensor-split`
  unset the estimate still models the engines' free-VRAM proportion.
  `--estimate --json` carries it as `balanced_split`; a card shortfall
  message names it, says whether it fits alone, names the value it
  replaces, and adds the `--fit` note only where `--fit` would otherwise
  act. Where the split does not fit alone, the offload count of SPEC 2.21
  is searched at that split and the message names both.
- New pure module core/balance.py, 29 tests of its own; the suite went from
  2039 to 2103 tests.
