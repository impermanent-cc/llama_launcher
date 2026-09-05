# llama_launcher: tasks

## Current phase

Branch chore/upstream-0-4-0-audit, complete and committed, held back from
merge and push on purpose: llama.cpp 0.4.0 was tagged 2026-09-04 but ghcr
has not yet published a server image carrying it (the floating :server tag
was still b10795 during the cycle), so the smoke below cannot run. When an
image past b10816 appears, run it, then merge with --no-ff and push. The
pre-push hook runs a full localci again on the way out.

## Open items

- [ ] The engine gate `setting.engine != "any" and setting.engine != engine`
      is written out three times: command_builder's emit loop,
      validation._is_active and vram.effective_ctx_size. One
      `accepts(setting, engine)` in settings_catalog would hold it once. The
      three agree today and a parity test pins two of them.
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
- [ ] configure_panel._member_estimates and _render_fit_line have no tests at
      all, so a future revert of the router fit readout would be silent.
- [ ] configure_panel calls self.current_profile() twice inside one
      fit_summary call on the debounced render path.
- [ ] No test asserts that an engine-gated build_catalog option carries a
      tooltip.
- [ ] video-timestamp-interval takes a minimum of 0 and nothing here
      establishes what upstream does with 0.
- [ ] setting_widgets.py:147 carries a doubled-hyphen prose separator and
      narrates history in a comment; one for the documentation cycle's prose
      sweep, along with RPC.md's non-ASCII at lines 143 and 163.

## Pending owner smokes

- [ ] Smoke the new flags against a real llama.cpp 0.4.0 image once ghcr
      publishes a server tag past b10816: a --dry-run profile carrying
      --kv-unified-per-slot, --no-reasoning-preserve and the --video-*
      flags, then a launch that reaches the ready state. The branch is not
      pushed until this passes.
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

- The periodic upstream re-audit against llama.cpp 0.4.0, tagged 2026-09-04.
  Verified first: the llama-server flag set at b10795, which carries every
  0.3.0 to 0.4.0 change to common/arg.cpp, is identical to
  tests/fixtures/llama_server_flags_b10711.txt, so no catalogued flag was
  renamed or removed and the fixture stands unchanged.
- build_catalog gates GGML_CUDA_PEER_MAX_BATCH_SIZE to ik_llama.cpp, which
  mainline deleted in 0.4.0 (PR 28177) and ik still defines.
- reasoning-preserve is deprecated and no-reasoning-preserve added, because
  0.4.0 enables preservation by default (PR 28174). The shared deprecated-row
  tooltip, its docstring and the Setting.deprecated field comment stopped
  claiming upstream retires every such flag and replaces it with --load-mode,
  which was only ever true of the load flags.
- A catalog-derived warning when a flag and its --no- twin both act, naming
  which one wins, including when one half arrives through raw args, where the
  existing dedup cannot see the collision.
- Four new mainline-only settings: kv-unified-per-slot and the three video
  flags. --spec-synth-len and --spec-synth-rates are excluded on purpose
  (SPEC.md section 4): upstream marks them benchmarking only and they falsify
  acceptance.
- vram.effective_ctx_size teaches the preflight the kv-unified-per-slot rule,
  and every single-node estimate path now shares it; validation warns when the
  slot count is unknowable.
- --n-cpu-ffn reaches the capability tiers, the RPC centralizing warning, the
  benchmark snapshot, the fit hint and RPC.md.
- Three engine-gate bugs found and fixed along the way, all the same shape:
  the pair warning, the preflight and the RPC centralizing warning each acted
  on a flag the chosen engine never receives. All three go through one
  predicate now.
- SPEC.md 2.12 and 2.13 were corrected during the cycle because the sentences
  as drafted were wrong, not the code: the n-cpu-moe and n-cpu-ffn pair is not
  symmetric, and the centralizing warning is engine-gated.
- The suite went from 1758 to 1796 tests.
