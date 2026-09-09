# llama_launcher: tasks

## Current phase

Idle: no cycle open. The last cycle, feat/configure-density-and-readout,
landed on 2026-09-09: the suggestion dot beside its editor and the tools
boxes in two columns, a finer --ctx-size ladder, the layer count on the
meta line with per-card layer ranges in the readout and a collapsible
Details section, --estimate printing the details with new JSON keys, and
bounded widths for the Environment column and the top bar fields. Next:
the owner's GPU smoke below, then a release commit on main that sets
pyproject to 0.2.0 and dates the CHANGELOG section, then the v0.2.0 tag
and the GitHub release, each on the owner's yes.

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
- [ ] The draft model's memory estimate uses the main model's micro-batch
      size and slot count, although llama.cpp builds the draft context with
      one sequence and its own batch; the pre-existing context fallback
      makes the same approximation.
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

- [ ] On the 5080 plus A2000 box, load the 27B dense profile and the
      35B-A3B profile and compare the readout's per-card layer ranges
      against the per-card model buffer lines of a Verbosity 4 launch (the
      boundary layer must match), the Details block's KV per 1024 tokens
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

- The suggestion dot sits 16 px wide directly after its editor with the
  row's stretch after it, and multiselect settings lay out in two columns,
  so the settings column's minimum width fell from 613 to 487 px offscreen
  and needs no horizontal scrollbar. Driven offscreen at 1600 px on the
  e2b profile with a screenshot of the Configure tab.
- --ctx-size presets: 1024 and the midpoints 12288, 24576, 49152, 98304 and
  196608 join the ladder.
- The Environment column is bounded to 420 to 640 px with the settings
  column absorbing the rest (measured: 640 from a 1400 px window on); the
  Name edit and the profile combo grow to 260 and 340 px and stop, with a
  stretch before the status label.
- The estimate carries a LayerLayout: each block layer's card from its KV
  placement, host layers, the output device, row split, per-card weight
  bytes over the layers the card holds (promoted host-layer bytes excluded),
  one layer's expert share (lowest expert layer, EXPS_REGEX, memoised per
  table and layer count) and the output tensor's size; a no-table model
  reports per-layer weights as unknown. estimate_memory takes a ctx
  override.
- fit_report prices the KV cost of the next 1024 tokens per device as the
  marginal cost at the profile's context, only with with_details, and
  carries the meta for the header facts.
- The meta line carries the layer count; each card line opens its bracket
  with the layer range ("layers 4 to 7 plus output", "layers 0 to 7, row
  split"), the RAM line names the host layers; a collapsible Details section
  (collapsed at start, plain text) holds the KV per 1024 tokens, per-layer
  weights with the expert share, the head, embedding and vocabulary counts,
  the sliding window and the output tensor; --estimate prints the block
  after the lines and --json carries n_layers, per-card layers and
  bytes_per_layer, ram layers, kv_per_1k and output_device. On the e2b
  profile: "layers 0 to 20" and "layers 21 to 34 plus output" at 60,40,
  KV per 1024 tokens 6 MiB, 24 and 35 MiB per layer, output 216 MiB.
- Two plan errors caught by the tests: the engine offloads the last block
  layers first (so -ngl 5 on eight layers holds 4 to 7 plus the output), and
  the default four slots share the context, so a sliding window binds only
  from 8192 tokens with a 1024 window.
- Docs: CHANGELOG [Unreleased], VRAM.md JSON keys and readout paragraph,
  README's --estimate clause, SPEC 2.24, 2.25, 2.38 to 2.42, ROADMAP Later
  (benchmark prompts, queued runs, the Speculative Decoding group width).
