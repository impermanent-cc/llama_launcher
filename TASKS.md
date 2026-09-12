# llama_launcher: tasks

## Current phase

Idle: no cycle open. fix/flag-availability-b10902 landed on 2026-09-12,
from an audit of ghcr.io/ggml-org/llama.cpp:server at build 10902 against the
catalog: SPEC 2.45, 2.46 and 2.47 are new, 2.12, 2.20 and 2.26 amended, and
--log-jsonl is recorded out of scope. Next: whatever the owner picks from
ROADMAP.md "Next"; the Gemma tied output tensor and the balanced split in the
sweep remain the two estimate items that measured runs already motivate, and
the open items below carry three follow-ups this cycle deliberately left,
the live-launch log warnings, the SPEC 2.46 load-mode refinement and the raw
verbosity route into the sweep.

## Open items

- [ ] Engine-gating the load-mode suppression leaves it with no observable
      positive case on either engine: mainline carries --load-mode but no
      longer catalogues the legacy pair, and ik carries the pair but never
      receives --load-mode, so only the negative branch is testable. Three
      sites hold the now-inert mechanism
      (command_builder._load_mode_suppression, router_preset,
      configure_panel._sync_load_mode_legacy). Keeping the gate
      is right while the tags can move again; removing the mechanism is a
      separate cleanup.
- [ ] Two comments in settings_catalog.py name test artefacts, which the house
      rule bans: one cites tests/fixtures/regen_ik_flags.sh and one names
      test_catalog_upstream_flags. Both predate the b10902 cycle and were left
      alone rather than growing that branch at its finish gate.
- [ ] Raw verbosity defeats the sweep the same silent way SPEC 2.26's refusal
      now blocks for --log-disable and --log-colors: services/sweep.py forces
      verbosity 4 because the buffer lines do not print below it, but a raw
      --verbosity 0, -lv 0 or --log-verbosity 0 wins as the later token, so
      the sweep names a winner on zero memory figures. Harder than the
      log-colors clause was: the test is "raw verbosity below 4" rather than
      a membership check, and --log-verbosity is uncatalogued so
      raw_arg_values returns nothing for it and the refusal needs token-level
      parsing. Needs a SPEC 2.26 sentence. Owner deferred it on 2026-09-12.
- [ ] tests/services/test_native.py's _await_exec returns silently when its
      5 second budget expires, so a starved execve leaves the test asserting
      against a pre-exec process and
      test_proc_state_survives_a_comm_containing_spaces_and_parens fails on
      its own "comm did not actually embed a paren" guard. Seen once under
      three concurrent suites on 2026-09-12, then five clean runs in
      isolation; native.py is untouched by that cycle. Raising a timeout
      error instead of returning would make the flake self-describing.
- [ ] The unexposed-flag list has no committed generator: the script that
      produces it lives in the cycle plan under docs/, which is untracked and
      deleted at the finish gate. Adding one line by hand is fine, a build
      that moves thirty flags is not. Land the snippet as
      tests/fixtures/regen_unexposed.py and cite it from regen_flags.sh.
- [ ] Nothing checks catalogued aliases against the capture: the covered set
      minus the capture is empty today, so a bogus alias would silently
      absorb a real new flag and keep the audit green. One assert locks it.
- [ ] After regen_flags.sh writes a capture for a new build, a forgotten
      FIXTURE edit in test_catalog_upstream_flags.py leaves the whole audit
      green against the old build. Asserting exactly one
      llama_server_flags_*.txt exists would catch it.
- [ ] unexposed_flags_mainline.txt is byte-stable under LC_ALL=C sort but
      nothing enforces it, so a regeneration under a UTF-8 collation would
      produce a spurious reordering diff. regen_flags.sh pins LC_ALL for the
      capture; the list has no equivalent guard.
- [ ] A headless `--launch` never shows a validation warning: the preflight in
      app.py:52-64 keeps only `i.level == "error"` and drops the rest, so the
      SPEC 2.46 legacy-flag notice reaches `--dry-run`, which prints every
      issue, and the GUI, which prompts, but not a scripted launch. Every
      warning the validator raises is invisible on that path, not just this
      one.
- [ ] The SPEC 2.46 suggestion ignores the load-mode already set: a profile
      with `{"mlock": True, "load-mode": "dio"}` is offered `mmap+mlock`, so
      one click replaces an explicit choice, and the warning advises
      --load-mode where it is already set and the legacy key was already
      inert through _load_mode_suppression. Clearing the legacy keys and
      leaving a non-default load-mode alone is the behaviour that reads
      right; 2.46 as written says otherwise, so this is a spec refinement.
- [ ] configure_panel.py:927 `_on_engine_changed` hides the mlock row without
      calling apply_model_caps(), so flipping the engine to llama.cpp in the
      form leaves the value set with no dot until the next caps refresh or a
      profile reload. Pre-existing; the engine-gated suggestion of 2.46 is
      what makes it reachable.
- [ ] capabilities._sug_legacy_load gates on accepts(CATALOG["mlock"]) alone
      while validation checks each key against its own tag, so if the two
      settings' engine tags ever diverge the suggestion would clear a
      still-live --no-mmap. Both should read per key.
- [ ] Retagging a setting's engine can make an assertion pass for a new
      reason without failing, and only mutation finds it. The b10902 retag of
      --mlock and --no-mmap left three such assertions, two of them fixed in
      place and one reduced, plus two that remain doubly satisfied:
      test_command_builder.py's test_setting_emits_follows_the_argv_rule and
      test_validation.py's
      test_is_active_matches_the_emit_rule_and_treats_zero_counts_as_idle both
      use mlock on a mainline profile as their "present but not emitted"
      example, which the engine gate now satisfies rather than load-mode
      suppression. Each still pins its own subject, so this is clarity rather
      than lost coverage; an ik profile is the only way to illustrate
      suppression now.
- [ ] validation.py:361 makes the same class of claim the 2.20 fix removed: on
      an ik profile with run-time-repack set and no load-mode, it appends "Your
      load-mode is mmap, which it overrides" because the read defaults to
      "mmap" while the catalog default is "auto", and the whole branch is
      ik-only, where --load-mode never reaches the launch. Both halves are
      wrong: the value and the relevance.
- [ ] tests/core/test_describe_relevance.py's override-tensor tests assert
      the representation rather than the behaviour: absence from the
      relevance map, not the hidden dot. ConfigurePanel._dot_state_for is a
      staticmethod and needs no QApplication, so asserting ("none", "") would
      survive a change to how a neutral tier is represented. The dense
      absence test also passes vacuously against a dense branch returning {},
      proved by mutation; the file kills that mutant through its neighbours,
      the test alone does not. One positive anchor in the same test would
      make it self-anchoring, and the third new test lacks the docstring its
      neighbours carry.
- [ ] The monitor and the MTP stats parse the log of a normal launch, so
      --log-disable, --log-colors on, or a raw --log-jsonl silently empties
      them; only the sweep protects itself (SPEC 2.26). A validation warning
      covering the live-launch surface as a whole is its own cycle with its
      own SPEC item.
- [ ] --mmproj-device is not modelled: the estimate charges the projector to
      --main-gpu whatever the flag says, and b10902 changed the upstream
      default from auto to following --device. Needs a vision model measured
      on the owner's box; there is no mmproj calibration record.
- [ ] The estimate does not follow an --override-tensor target card:
      parse_overrides records CPU or card, not which card, so the output
      layer stays charged to the last card in the split even when
      'output\.weight=CUDA0' moves it. The readout's layer ranges and the
      logits term follow the split, not the override.
- [ ] --device reordering is not modelled either way: --tensor-split indexes
      the device list, so --device CUDA1,CUDA0 with -ts 20,46 gives CUDA0 the
      larger share and the output layer, while the estimate reads the split
      positionally against physical cards and mirrors the result.
- [ ] llama.cpp clamps the speculative sequence count to zero when the
      target architecture is not in its rollback list (qwen35, qwen35moe,
      qwen4exp, deepseek4, nemotron_h, nemotron_h_moe, lfm2, lfm2moe,
      bailingmoe3, kimi_k3 as of b10818) and speculates through
      checkpoints instead, so a Falcon-H1, Jamba or Mamba2-class hybrid
      under a rollback spec-type is charged slots x depth state units it
      never allocates. The estimate never consults meta.arch for this;
      narrow today, since such heads barely exist for those archs.
- [ ] The host buffer band has little headroom: the E2B cpu-only record
      reads 2.468 times the measured figure against the 2.5 ceiling, the ik
      fit record 2.428 and the two-slot drafted 27B record 2.289, so a
      host-term change trips one of those three first.
- [ ] The main window's minimum width is about 1032 px, from the top
      bar's Name and profile picker minimums (160 and 200 px) plus six
      buttons, so a 1024 px display no longer fits the window; the owner
      accepted it on their display on 2026-09-10. Lowering
      NAME_EDIT_BOUNDS[0] or letting the buttons collapse would fix it;
      ENV_COLUMN_MIN (420) is inert while that minimum stands.
- [ ] The readout's layer ranges follow the KV cache placement while a model
      with no tensor table parks its whole weight blob at the output
      position, so a partial offload of such a model names host layers with
      "weights 0.0" behind them (VRAM.md's fallback). The Details block
      already says the per-layer weights are unknown there.
- [ ] A card holding a draft model or projector but no main-model layer reads
      "no layers" beside non-zero weights and KV per 1024 tokens; SPEC 2.24
      says the ranges cover the main model, and the line does not.
- [ ] memory_fit.render_lines and render_details guard against an estimate
      with no layout or an empty kv_per_1k, which fit_report never produces;
      to_json's two card zips stay strict=False on the same equal-length
      invariant _kv_per_1k now zips strictly.
- [ ] The guard allowlist lives three times: tests/guard/conftest.py and
      two grep steps in ci.yml's sanity job, which skip only the lock files,
      anchor on exact names and never fail on a stale entry as the pytest
      guard does. Replacing the grep steps with `pytest tests/guard` (the
      test job already runs it) would leave one list; design call, since the
      sanity job runs without Python.
- [ ] The Speculative Decoding group is now the widest in the settings
      column (436 px offscreen), from bool rows whose checkbox text repeats
      the flag beside the row label (ROADMAP Later).
- [ ] --device is not modelled: every visible card is counted and gets the
      per-card overhead even when the launch excludes it (VRAM.md, known
      limits).
- [ ] ik_llama.cpp layer mode fills cards by cumulative bytes rather than
      layer index, so per-card weights drift on uneven (cpu-moe) layers;
      placement.distribute's engine parameter is unused.
- [ ] Split-model parts hardlinked under two names with a disagreeing
      split.count key count twice (malformed layout only); the Configure
      cache stamps only the first part.
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
- [ ] A standalone draft that is itself a hybrid would be over-charged: a
      draft is priced a cache on every layer its file carries, recurrent
      ones included, and no measurement covers such a model. Over-charging
      is the safe direction for a fit check.
- [ ] An ik_llama.cpp draft adds no state units, since spec-draft-n-max is
      mainline-only while ik carries its own draft-params row.
- [ ] The ik calibration record's card 0 KV lower bound clears by a few KiB
      only through the log-precision allowance; a second ik measurement
      would settle whether the one-layer slack is enough.
- [ ] Owner, GitHub: ask GitHub Support to garbage-collect the unreachable
      objects left by the 2026-08-31 history rewrite. The unredacted
      router.png blob (91a946f) was still fetchable by exact SHA on
      2026-09-02 and the repository is public; nothing records the request
      as filed.
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
      walk with the parity tests covering the untied fixture only, and the
      search-entry visibility closure is a third copy of the
      getWidgetPosition/isRowVisible idiom.
- [ ] configure_panel.cached_probe does not adopt a pending
      _fit_probe_result, so a launch clicked inside the 150 ms poll gap
      re-probes with a fresh reading already in hand; the tests pin this
      shape, so changing it is a design call.
- [ ] The shortfall message shell-quotes the --override-tensor suggestion
      with shlex.quote, so a value carrying no shell metacharacters would
      render bare; every generated value carries regex metacharacters
      today, so SPEC 2.21's single quotes hold in practice.

## Pending owner smokes

- [ ] On the 5080 plus A2000 box against a b10902 image, launch a real
      profile and confirm the server reaches ready with no unknown-argument
      line in the log. This cycle is catalog work, so it is verified by
      running, not by the suite.
- [ ] On a dense model, confirm the --override-tensor row shows no dot and
      accepts a value, then launch with `-ot 'output\.weight=CUDA0'` and read
      the per-card buffer lines at verbosity 4 to confirm the output tensor
      moved to card 0. The estimate will still charge it to the last card in
      the split; that gap is an open item above.
- [ ] Load a profile that still carries mlock, confirm the settings-route
      warning appears and that clicking the load-mode suggestion clears both
      the key and the warning in one go.
- [ ] Put `--mlock` in a profile's raw args on a b10902 image and confirm the
      raw-route warning appears and says the launch fails on an image that
      new, then confirm it does fail if launched.
- [ ] Run an offload sweep and confirm it still records memory figures per
      point, and that a profile with `--log-disable` in its raw args is
      refused rather than sweeping to a winner with zeroes.
- [x] Screenshots: checked, no re-shoot needed. The two rows this cycle
      removes from a mainline form sit immediately below config.png's
      captured fold, which ends at --lazy-mode, so the captured surface is
      unchanged.
- [x] Re-shoot assets/screenshots/config.png and build.png on the 5080 plus
      A2000 box, where the group titles had the ampersand swallowed. Done
      2026-09-11 along with router.png: "Model & Context", "GPU & Memory"
      and "CPU & ISA" render as ampersands.
- [x] Run the 27B profile at `--parallel 2` with the MTP draft on. Done
      2026-09-10, the fourth run in
      DevDocs/llama_launcher/calibration-2026-09-10/verb4-27b-and-35b.md.
      `llama_memory_recurrent: size = 897.75 MiB (2 cells, 64 layers, 2
      seqs 2 rs_seq)`: the rule is `slots x (depth + 1)`, six units, and
      the estimate's `slots + depth` reads 299.25 MiB low (the first open
      item). The draft's KV buffer read 352.00 MiB at 45056 cells times 2/2
      seqs: the draft keeps the full context, divided per slot like the
      main model's, so the estimate's full-context pricing holds. The
      launcher's draft-mtp warning against --parallel above 1 fired and
      was ignored; the server ran both slots with the draft (the second
      open item).
- [x] Re-shoot assets/screenshots/bench.png on the 5080 plus A2000 box now
      that the columns are formatted. Done 2026-09-10: the history table
      reads 32.9, 43.9 and 3.04, right-aligned.
- [x] On the 5080 plus A2000 box, compare the readout's per-card layer
      ranges, KV per 1024 tokens and per-layer weight against Verbosity 4
      launches of the 27B and 35B-A3B profiles. Done across the 2026-09-06
      and 2026-09-10 logs: the per-card model buffers match the estimate to
      the byte on both profiles at two different splits of the 27B (60,40
      and 43,23), which pins the boundary layer and the per-layer weight;
      the KV matches exactly per card at two context sizes of the 27B
      (32768 and 90112) and on both cards of the 35B, which pins the KV
      per 1024 tokens.
- [x] Confirm on your display that the settings column shows no horizontal
      scrollbar, that the Environment column and the top bar fields stop at
      their maxima, and that the 1032 px window minimum is acceptable.
      Confirmed by the owner on 2026-09-10.
- [x] Take a profile that is over budget on one card on the 5080 plus
      A2000 box, apply the suggested `--tensor-split` from
      `--estimate --json`, launch at Verbosity 4 and check the per-card
      model buffer lines against the predicted boundary layer. Done by the
      owner before 2026-09-10: the suggestion is what put the 27B on the
      layer-count split 43,23, and that launch is the 2026-09-10 27B
      calibration record, whose model buffers match the estimate to the
      byte on both cards.
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
- [x] Re-read the g31b_ud4k container's log once the run has finished.
      Dropped 2026-09-10: the owner deleted the 31B profile because it ran
      poorly on the box, so the 2026-09-06 paste stays model buffers only.
- [x] Measure one dense non-hybrid model on the 5080 plus A2000 box. Done
      2026-09-06 with Gemma 4 12B (dense, MTP draft), 26B-A4B (MoE) and 31B
      (dense, model buffers only), mainline b10818 at Verbosity 4 with
      --swa-full; print_info blocks, buffer lines, commands and estimate
      JSON in DevDocs/llama_launcher/calibration-2026-09-06. Weights are
      exact on every card; KV reads 1.9x to 2.2x high and compute reads low
      on card 0, see the open items. Not yet added as records: the KV rule
      fails until the reader knows the sliding-window head size.
- [x] On the 27B dense profile, run a sweep whose "from" sits a few counts
      below the prefilled start so the first point is too small: confirm it
      records as failed with its log line in the tooltip and the sweep
      continues. Confirmed by the owner on 2026-09-10.
- [ ] On that same sweep, confirm the winner is marked and Apply writes the
      count into the profile.
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

- `--mlock` and `--no-mmap` are ik-only: mainline rejects both from build
  10902, and the engine gate keeps them out of the form and out of argv on
  every path (SPEC 2.45).
- A profile still carrying either warns, with the wording matching the route,
  since a settings value is dropped while a raw spelling reaches the launch
  and fails; a settings value also offers the matching `--load-mode`
  (SPEC 2.46).
- `--override-tensor` is unmarked on a dense model rather than "not
  applicable", and stays a tuning knob on a MoE one (SPEC 2.12).
- The RAM shortfall wording treats a legacy `--mlock` as a lock only where
  the engine accepts it (SPEC 2.20).
- The load-mode suppression is engine-gated at all three sites, so an ik
  profile with a leftover load-mode keeps the pair and its checkboxes,
  including after an in-place engine switch.
- The sweep carries the log format its parser reads and refuses raw args that
  hide it (SPEC 2.26).
- The mainline flag fixture tracks b10902 and gains a committed list of the
  155 flags the launcher does not use there, so regenerating the capture is
  the audit and a new upstream flag fails the suite naming itself (SPEC 2.47).
- `--direct-io` references are gone; the `--mmproj-device` tooltip matches
  b10902's "follows --device" default.
