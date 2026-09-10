# llama_launcher: tasks

## Current phase

Idle: no cycle open. The last cycle, fix/draft-cache-and-state-cells,
landed on 2026-09-10: the memory estimate now prices a draft model's cache
and every model's recurrent state from what llama.cpp allocates, measured
against one Verbosity 4 log of three runs (SPEC 2.17, 2.19 and 2.32; the
log, the estimates and a term by term table are in
DevDocs/llama_launcher/calibration-2026-09-10). Both 27B runs now reproduce
the logged weights, KV and recurrent state exactly on both cards. Next: the
owner's GPU smokes below, the first of which settles the state cell rule,
then a release commit on main that sets pyproject to 0.2.0 and dates the
CHANGELOG section, then the v0.2.0 tag and the GitHub release, each on the
owner's yes.

## Open items

- Owner: the main window's minimum width rose from about 843 to 1032 px,
      driven by the top bar's Name and profile picker minimums (160 and
      200 px) plus six buttons, so a 1024 px display no longer fits the
      window. Lowering NAME_EDIT_BOUNDS[0] or letting the buttons collapse
      fixes it; ENV_COLUMN_MIN (420) is inert while that minimum stands.
- [ ] The readout's layer ranges follow the KV cache placement while a model
      with no tensor table parks its whole weight blob at the output
      position, so a partial offload of such a model names host layers with
      "weights 0.0" behind them (VRAM.md's fallback). The Details block
      already says the per-layer weights are unknown there.
- [ ] A card holding a draft model or projector but no main-model layer reads
      "no layers" beside non-zero weights and KV per 1024 tokens; SPEC 2.24
      says the ranges cover the main model, and the line does not.
- [ ] Group box titles carrying "&" ("Server & Tools", "Model & Context")
      render the ampersand as a mnemonic underscore on some platforms
      (seen offscreen); QGroupBox titles need "&&".
- [ ] memory_fit.render_lines and render_details guard against an estimate
      with no layout or an empty kv_per_1k, which fit_report never produces;
      to_json's bytes_per_layer indexes card_layer_bytes by card["index"]
      inside a positional zip, and _kv_per_1k zips with strict=False where
      strict=True would surface a length mismatch.
- [ ] The Speculative Decoding group is now the widest in the settings
      column (436 px offscreen), from bool rows whose checkbox text repeats
      the flag beside the row label (ROADMAP Later).
- [ ] placement.layer_index is a public alias of _layer_of with one caller
      in vram; renaming _layer_of would remove the pair.
- [ ] --device is not modelled: every visible card is counted and gets the
      per-card overhead even when the launch excludes it (VRAM.md, known
      limits).
- [ ] ik_llama.cpp layer mode fills cards by cumulative bytes rather than
      layer index, so per-card weights drift on uneven (cpu-moe) layers;
      placement.distribute's engine parameter is unused.
- [ ] Split-model parts hardlinked under two names with a disagreeing
      split.count key count twice (malformed layout only); the Configure
      cache stamps only the first part.
- [ ] src/llama_launcher/ui/widgets/setting_widgets.py:147 carries a
      doubled-hyphen prose separator and narrates history in a comment; one
      for the documentation cycle's prose sweep, along with RPC.md's
      non-ASCII at lines 143 and 163.
- [ ] The RAM estimate does not model llama.cpp's CPU_REPACK buffer: on a
      CPU-only launch the server keeps a repacked second copy of the
      weights it runs on the CPU (1.2 GiB beside the 2.5 GiB mmap of the
      e2b model on 2026-09-06, measured 3.8 GiB against 3.4 estimated).
      The same applies to expert layers kept in RAM by the offload knobs.
- [ ] The host multiplier in COMPUTE_TERMS also scales the attention-scores
      term, which no calibration record covers since every record ran with
      flash attention on.
- [ ] The checkpoints term charges the --ctx-checkpoints maximum (32 by
      default) times the recurrent state per slot to RAM, although the
      server creates checkpoints on demand. Confirmed 2026-09-10 to be
      exactly 32 times the state on both profiles (9776 MiB of the 27B's
      11.4 GiB RAM estimate, 8040 MiB of the 35B's 9.6 GiB), with no
      load-time allocation answering it in either log; the 35B run does
      carry "speculative decoding will use checkpoints", so the
      checkpoints are real and only their size is unmeasured. Watch a long
      session on the 35B-A3B for RAM growth into it.
- [ ] Multi-head latent attention (deepseek2 and kin) is priced at the
      header's per-head key and value lengths, far above the engine's
      latent cache; reading kv_lora_rank plus the rope dimension would fix
      it (SPEC out of scope).
- [ ] With no tensor table, an --override-tensor rule that promotes the
      output tensor to a card while -ngl leaves the output layer in RAM
      still sends the logits term to the host buffer; the device flag wins
      over the override.
- [ ] The 35B-A3B's host weight estimate reads 1372.95 MiB against a logged
      host model buffer of 515.31 MiB (2026-09-10), while the 27B's host
      weights are exact. Nothing yet explains the difference.
- [ ] recurrent_layer_mask treats any layer with a non-zero per-layer
      feed-forward width as not recurrent, so a hybrid whose recurrent block
      also carries an MLP would be charged no state (SPEC 2.32 as written).
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
- [ ] The draft model's memory estimate uses the main model's micro-batch
      size, although llama.cpp builds the draft context with its own batch.
      The slot half of this is fixed: a draft's cache is priced at one
      sequence.
- [ ] An MTP draft reserves a large compute buffer for a small weight (554
      MiB on card 1 of the 2026-09-10 run, 3320 MiB on card 0 of the
      gemma4-assistant run) and no term covers it.
- [ ] A COMPUTE_TERMS refit is owed, and the three 2026-09-10 records name
      the figures it must reach: the 35B's card 0 compute at 0.883 and its
      host buffer at 0.833, both reading low, and the two-slot 27B's host
      buffer at 861.84 MiB against 197.13 measured, 4.37 times high. Each
      record marks those figures pending, so the bands hold everywhere else
      and the fitting script leaves them out of the region it fits.
- [ ] The drafted run logs two output buffers, the main model's and the
      draft's at 0.95 MiB each, while the estimate charges one; the record
      marks its output figure pending.
- [ ] A 27B profile with no draft loaded is charged the weights of the
      file's unused blk.64.nextn.* tensors, 334.75 MiB on the card holding
      the tail (5883.07 MiB with the draft on against 5548.32 with it off,
      2026-09-10). Placement counts every block tensor in the table whatever
      nextn_predict_layers says; extending the exclusion to placement is the
      natural follow-up to this cycle.
- [ ] The estimate prices a draft at the main model's context when
      ctx-size-draft is unset, while llama.cpp server appears to build the
      draft context at n_ctx divided by n_parallel. The only drafted
      measurement ran one slot, so it cannot show the difference; the
      pending smoke settles it.
- [ ] A standalone draft that is itself a hybrid would be over-charged: a
      draft is priced a cache on every layer its file carries, recurrent
      ones included, and no measurement covers such a model. Over-charging
      is the safe direction for a fit check.
- [ ] An ik_llama.cpp draft adds no state cells, since spec-draft-n-max is
      mainline-only while ik carries its own draft-params row.
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
- [ ] A mounted draft model or projector whose file is absent yields
      (None, None) from inspect_file exactly like an unmounted one, so it
      counts zero bytes with no message; SPEC 2.35 covers only the unmounted
      case. Spec gap to grill.
- [ ] placement's per-layer sums memo is shared across threads: the sweep
      controller's estimate closure runs estimate_memory on its QThread while
      the UI thread renders. Each dict operation is atomic, so the worst case
      is a duplicate walk. The memo pins up to 64 tensor tables alive by
      strong reference, and placement._compute_family repeats the "_exps"
      name scan memory_fit.is_moe does, since the layering keeps them apart.
- [ ] tests/core/test_calibration.py's layer_kv_bytes re-derives vram's
      per-layer KV rule (window tokens, per-layer heads, batch clamps); a
      public vram.kv_layer_sizes(meta, settings, engine, ctx, ubatch) used by
      _model_part and the test would be one source. Its window branch is
      unreachable until a carded sliding-window record exists (the only
      sliding-window record is CPU-only), and the slack prices KV alone while
      the band compares kv plus state, safe at today's state sizes.
- [ ] core.sweep.parse_prompt_sizes accepts 0 and negative sizes, as the
      parse it replaced did; a token > 0 guard is a one-liner if the
      benchmark client cannot use them.
- [ ] benchmark_controller.sweep_prefill(profile=None)'s default has no
      production caller (three tests call it bare); the memo reset at the
      top of configure_panel._refresh_fit_line is defensive and no test
      observes it.
- [ ] A deprecated setting's row label keeps its inline palette(mid) span
      colour on top of the search bar's jump tint, so "*deprecated" stays
      grey on the highlight fill.
- [ ] On every render of an over-budget profile the sweep prefill re-runs
      smallest_fitting_offload, the same bisection the memoised FitReport
      just ran inside its messages. Carrying the profile search's (key,
      value) on FitReport, None where the balanced split made the search
      unnecessary, would let sweep_prefill read it from the memo.
- [ ] The offload search's --override-tensor candidate string is the only
      channel carrying the layer count into placement.place, which recovers
      it by reverse-parsing the string with _COUNT_OVERRIDE_RE; a
      semantics-preserving change to the builder would silently fall back to
      one tensor walk per bisection step (a test now pins the ik path).
      Passing a structured (count, regex) through estimate_memory and place
      would drop the parse.
- [ ] Cleanup: placement._compute_layer_sums repeats _place_walk's tensor
      walk with the parity tests covering the untied fixture only;
      memory_fit._messages carries the balanced-differs predicate twice;
      the search-entry visibility closure is a third copy of the
      getWidgetPosition/isRowVisible idiom; SearchEntry.key is written and
      never read.
- [ ] configure_panel.cached_probe does not adopt a pending
      _fit_probe_result, so a launch clicked inside the 150 ms poll gap
      re-probes with a fresh reading already in hand; the tests pin this
      shape, so changing it is a design call.
- [ ] The shortfall message shell-quotes the --override-tensor suggestion
      with shlex.quote, so a value carrying no shell metacharacters would
      render bare; every generated value carries regex metacharacters
      today, so SPEC 2.21's single quotes hold in practice.

## Pending owner smokes

- [ ] Run the 27B profile at `--parallel 2` with the MTP draft on. It
      settles two open questions at once. The state cell rule: three
      candidates fit both measured runs, `slots + depth` predicting 4 cells
      (598.50 MiB of state), `max(slots, depth + 1)` predicting 3 (448.88
      MiB) and `slots x (depth + 1)` predicting 6 (897.75 MiB); read
      `llama_memory_recurrent: size` and report which. The draft context:
      the draft's KV buffer reads 352.00 MiB if the draft keeps the full
      context and 176.00 MiB if llama.cpp divides it by the slot count.
      The estimate implements `slots + depth` and the full context.
- [x] Re-shoot assets/screenshots/bench.png on the 5080 plus A2000 box now
      that the columns are formatted. Done 2026-09-10: the history table
      reads 32.9, 43.9 and 3.04, right-aligned.
- [ ] On the 5080 plus A2000 box, load the 27B dense profile and the
      35B-A3B profile and compare the readout's per-card layer ranges
      against the per-card model buffer lines of a Verbosity 4 launch (the
      boundary layer must match; the 2026-09-10 log settles this half, the
      model buffers matching the estimate to the byte on both profiles), the Details block's KV per 1024 tokens
      against the KV growth between two context sizes in the log, and the
      per-layer weight against one layer moved by --tensor-split. Also
      confirm the settings column shows no horizontal scrollbar at your
      usual window size, the Environment column and the top bar fields stop
      at their maxima, and the 1032 px window minimum is acceptable on your
      display.
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
- [x] Re-run `--estimate --json` for the 27B dense and 35B-A3B profiles on
      the 5080 plus A2000 box and compare against the measured figures. Done
      2026-09-10 against one Verbosity 4 log carrying both profiles, filed
      with the estimates and a term by term table in
      DevDocs/llama_launcher/calibration-2026-09-10. Weights are exact on
      every device on both profiles, the draft's own buffers included, and
      the 35B's KV and recurrent state are exact on both cards. The 27B's
      KV reads exactly twice the truth and its recurrent state 32 percent
      under; see the open items below. Not done: a second sweep per profile
      with the measured against estimated columns.
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

- A draft model's KV cache is sized over the layers its own tensor table
  carries rather than the layer count its header declares, each priced at
  the index its tensor names give, at one sequence whatever --parallel
  says, and at f16 unless its own cache-type rows say otherwise. Every
  layer a draft's file carries holds a cache, whatever the full-attention
  pattern of the header it shares with the main model says. A draft adds no
  recurrent state.
- The trailing multi-token-prediction positions the header's
  nextn_predict_layers names hold neither a KV cache nor recurrent state in
  the model that declares them, which took the 27B from 49 charged
  recurrent layers to 48.
- Recurrent state is charged once per state cell, the slots plus the
  speculative depth a loaded draft asks for, while the checkpoints term
  stays per request slot. On the 27B that moves the checkpoints from 9776
  MiB to about 4.68 GiB.
- The three 2026-09-10 runs are calibration records. Records may mark
  individual figures pending, which keeps those out of their bands while
  every other figure stays pinned, and the fitting script leaves them out
  of the region it fits. Without that last part the new records left the
  fitter with an empty feasible set.
- Both 27B records reproduce the measured KV plus state at a ratio of
  1.0000 per card; the drafted run's card 1 reads 1614.27 MiB estimated
  against 1614.27 measured.
- The 2026-09-06 27B record is reconciled onto the 65-position header the
  file actually declares, and each record now carries its own logged free
  VRAM rather than the first run's.
- Docs: CHANGELOG [Unreleased], VRAM.md's recurrent-state and draft
  paragraphs, SPEC 2.17, 2.19 and 2.32.
