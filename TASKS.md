# llama_launcher: tasks

## Current phase

Idle: no cycle open. feat/prism-engine (started 2026-09-17) added PrismML's
llama.cpp fork as the third engine, `prism_llama.cpp`: SPEC 2.48 to 2.55 are
new and 2.1, 2.2, 2.26 and 2.45 amended. The cycle's two owner smokes below
are still open; the first is the fork's first calibration record. Next:
whatever the owner picks from ROADMAP.md "Next", or the smokes' findings.

## Open items

- [ ] The load-mode suppression has its positive case again on PrismML
      llama.cpp, which receives --load-mode and the legacy pair both, so
      the three sites (command_builder._load_mode_suppression,
      router_preset, configure_panel._sync_load_mode_legacy) are live
      there and untested in that direction: a PrismML profile with a
      non-default load-mode should grey out and drop --mlock and --no-mmap.
      Pin it with a test.
- [ ] The RPC pool launch appends --rpc directly, but the mainline fixture
      image (server-b10902) is built without RPC, so the flag is in neither
      the fixture nor the structural table and the audit cannot see a rename;
      the fork's fixture, from a tarball with RPC built in, lists it as
      unexposed instead. Capture mainline from an RPC-capable build or record
      --rpc structurally with a note.
- [ ] Headless --launch and --dry-run call validate() without a GGUF read,
      so the wrong-engine warning of SPEC 2.52 is GUI-only; a PrismML file
      launched headless under a mainline engine fails at model load with no
      preflight line.
- [ ] The card shortfall note in memory_fit names "llama.cpp" in prose where
      a PrismML profile now sees the same note; wording only, and only if
      engine-named readouts are wanted (ENGINE_LABELS exists).
- [ ] The Image field hint for PrismML and the Detect dialog's no-image text
      say the same thing in two places (configure_panel.PRISM_IMAGE_HINT and
      launch_controller.detect_image); only "Build tab" is asserted in both.
- [ ] test_settings_catalog pins len(PRISM_LACKS_FLAGS) == 6 as a bare count,
      which the fixture audit already derives; and the parametrised bucket
      test in test_catalog_upstream_flags asserts a "llama.cpp"-only bucket on
      MAINLINE_KEYS, which holds only while PRISM_LACKS_FLAGS is non-empty.
      Both fail confusingly the day the fork catches up with mainline.
- [ ] gguf.prism_quant reports the first fork-only type in a mixed tensor
      table; no published file mixes PQ2_0 and PTQ1_0, and the docstring does
      not state the tie-break.
- [ ] build_panel.py carries " -- " prose separators at three sites
      (documentation cycle, with the rest of the doubled-hyphen guard work).
- [ ] GGML_CUDA_HOPPER_Q1 needs GGML_CUDA_CUTLASS_DIR, which the build catalog
      does not carry, so a Hopper build of the fork must supply it through raw
      defines; either add the option gated to the fork or say so in the
      tooltip.
- [ ] The wrong-engine and kv-mean-center warnings sit under the non-router
      branch of validate(), so a router profile whose member carries PQ2_0 or
      PTQ1_0 weights relies on the member profile's own preflight.
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
      logits term follow the split, not the override. Measured 2026-09-12
      (DevDocs/llama_launcher/calibration-2026-09-12): the move is real
      and worth 1122.37 MiB on the 27B, so the CUDA0 estimate reads that
      much low under the override and CUDA1 the same high, while every
      other term still lands; the run still fitted because the margin
      covered it. When this lands, the walk already
      matches with re.search exactly as llama.cpp's regex_search does, so
      `output\.weight` must also move every blk.N.attn_output.weight to
      the target card; the anchored `^output\.weight` is the pattern that
      moves the output tensor alone, and any launcher text that suggests
      the pattern should say so.
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

- [ ] On the 5080 plus A2000 box (CUDA 13): from the Build tab, a container
      build of the PrismML engine from the `prism` branch with CUDA 13 builder
      and runtime images typed into the image fields; Detect then lists the
      `prism-custom` image; a container launch of
      Ternary-Bonsai-2-27B-PTQ1_0.gguf at verbosity 4 reaches ready and a
      completion returns coherent text (the model card wants temperature 1.0,
      top-p 0.95, top-k 20 for thinking mode). File the log and the
      `--estimate` JSON in DevDocs/llama_launcher as the fork's first
      calibration record, and compare the model buffer lines with the
      estimate's 5.94 GB of weights: this is the first measurement of a
      PTQ1_0 file and of the qwen35 hybrid state terms on this engine.
- [ ] Trailing, optional: the same from the Build tab's native target, a
      cmake build against the CUDA 13 toolkit, pointed at from a native
      profile.
- [ ] Screenshots: config.png's captured fold includes the Engine row, which
      gained a third combo entry and a suggestion dot beside it, and build.png
      shows the Build tab's engine combo. Re-shoot both on the GPU box once the
      first smoke has a working image, or confirm the fold hides the change.

## Done this cycle

- PrismML's llama.cpp fork is the third engine, stored as `prism_llama.cpp`
  and shown as "PrismML llama.cpp" in Configure and Build; build outputs are
  tagged `prism-custom` (SPEC 2.48, 2.55).
- A setting's engine gate is a set: `any`, one name, or a tuple. The fork
  receives mainline's catalog minus the six flags mainline added after build
  10685, plus `--mlock`, `--no-mmap` and its own `--kv-mean-center`, which
  warns when cache-type-k is not q4_0 on either route (SPEC 2.2, 2.45, 2.49).
- The fork's flag fixture is captured from the release tarball by
  `regen_prism_flags.sh` and audited in the three directions of 2.47, with a
  six-line unexposed list of its own (SPEC 2.50).
- The GGUF reader sizes PQ2_0 and PTQ1_0 at their block size and names file
  types 41, 141, 142 and 143; the owner's 27B PTQ1_0 file now reads 5.94 GB
  instead of 50 GiB, matching its data section byte for byte (SPEC 2.51).
- A PQ2_0 or PTQ1_0 model under mainline or ik gets a preflight warning and
  a dot beside Engine that switches the engine alone (SPEC 2.52).
- The estimate, the fit gate and the sweep's verbosity treat the fork as
  mainline; a whole-estimate equality test pins it (SPEC 2.26, 2.53).
- No default image for the fork: the Image field says so and points at the
  Build tab on every path, including profile load, and Detect matches
  `prism` (SPEC 2.54).
- The Build tab knows the fork's repository, `prism` ref, mainline's CMake
  options and `GGML_CUDA_HOPPER_Q1` (SPEC 2.55).
- README, CHANGELOG, VRAM.md and AGENTS.md name the third engine.
