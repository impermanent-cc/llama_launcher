# llama_launcher: tasks

## Current phase

Idle: no cycle open. The offload sweep cycle (feat/offload-sweep, SPEC.md
2.6 and 2.26 to 2.31) landed on main on 2026-09-06. The two owner smokes
that need the 5080 plus A2000 box are listed below, the sweep one first.

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
- [ ] tests/ui/test_fit_readout.py keeps a dead inspect_file patch and a
      stale docstring; tests/core/test_purity.py's enhancement-module test
      is a strict subset of the whole-core scan.
- [ ] Three LaunchController calls into the panel's private
      _cached_meta_weights; promote it to a public method.
- [ ] No test asserts that an engine-gated build_catalog option carries a
      tooltip.
- [ ] video-timestamp-interval takes a minimum of 0 and nothing here
      establishes what upstream does with 0.
- [ ] setting_widgets.py:147 carries a doubled-hyphen prose separator and
      narrates history in a comment; one for the documentation cycle's prose
      sweep, along with RPC.md's non-ASCII at lines 143 and 163.
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

## Pending owner smokes

- [ ] Run a sweep on the 5080 plus A2000 box on the 27B dense profile with
      the prefilled range: confirm each point launches, benchmarks, stops
      and removes its container, that the winner is marked and Apply writes
      the count, and that the measured columns are within a few hundred
      MiB of the estimate. Then lower "from" a few counts below the
      prefilled start so the first point is too small, confirm it records
      as failed with its log line in the tooltip and the sweep continues;
      report the table.
- [ ] Calibrate the compute constants on the 5080 plus A2000 box: set the
      profile's Verbosity to 4 first (llama.cpp 0.4.0 prints no memory
      table below that), then follow VRAM.md's procedure with the
      two-card 27B dense profile from the
      screenshot (tensor-split 60,40, ctx 98304, q8_0 KV) and one MoE
      profile; compare --estimate --json per card against the server's
      exit-time memory breakdown and report the four numbers per profile so
      COMPUTE_TERMS and CARD_OVERHEAD_BYTES can be refitted. Check the RAM
      line too: on a 262144-token vocabulary the host output buffer alone is
      about 2 GiB and dominates it.
- [ ] Launch a real profile and confirm the four-line readout, its tooltip
      and the launch dialog on KDE/Wayland; try an over-budget context to see
      the --fit note and the suggested offload count.
- [ ] Smoke an ik_llama.cpp profile with --fit on: the command carries bare
      --fit and the readout shows the ik note on a MoE model.
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

- An offload sweep in the Benchmark tab: one detached container per
  --n-cpu-ffn (dense) or --n-cpu-moe (MoE) count over a range the memory
  estimate prefills, each waited for on /health, its load-time model, KV
  and compute lines read from the container log, benchmarked with the
  panel's settings, then stopped and removed. Failed points record the last
  log line and the sweep continues; Cancel ends it after the current point
  and leaves the stored sweep untouched; a server that dies at load fails
  its point as soon as the container stops; closing the window removes a
  running sweep container on the way out.
- A pure core module (knob, counts, log parser, winner), a runner service
  with injectable probes, a per-profile sweep store beside the benchmark
  history, and the panel's sweep row and table with the winner marked and
  Apply writing the count into the profile.
- The sweep refuses router, native, RPC and remote-node profiles, a profile
  with no model or no prompt sizes, an engine without the knob, raw
  arguments carrying the knob's flag, a running instance of the profile or
  of a previous sweep container, and a running benchmark or sweep; a
  benchmark refuses to start during a sweep. The stored sweep of the loaded
  profile is shown when the profile loads.
- The estimated figure beside each measured card total is model plus KV
  plus compute, without the per-card overhead the log never reports.
- The suite went from 1933 to 1994 tests.
