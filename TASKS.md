# llama_launcher: tasks

## Current phase

Idle: no cycle open. Two branches await the owner's merge, 2026-09-03:
chore/workflow-stamp (the workflow standard, light touch) and
chore/ruff-gate on top of it (SPEC.md 3.5: the standard's ruff block, CI
lint job, autofix and reformat of 194 files, 47 findings fixed by hand,
the ruff gate green with 1756 tests passing). Merge with --no-ff in that
order, or merge chore/ruff-gate alone since it contains the stamp.

## Open items

- [ ] main_window.py re-exports eight service names (`import x as x`, one
      statement each) only so tests can patch `mw.<service>`; about 112
      patch sites in 12 test files. Patching the service modules directly
      (as tests/ui/conftest.py already does) would let the re-exports go.
- [ ] ruff is a PATH tool, not a dev dependency, so a contributor with a
      different ruff gets different results from the AGENTS.md lint row
      than CI. The workflow standard's block lacks required-version; once
      it carries one, a mismatch fails loudly here too.
- [ ] Documentation cycle to take README.md, CHANGELOG.md and RPC.md off
      the guard allowlist and to reword the legacy ` -- ` comment
      separators (about 60 lines), then set CHECK_DOUBLE_HYPHEN to True.
- [ ] docs/ is ignored, so tracked cycle plans under docs/plans need the
      ignore rule relaxed (`!docs/plans/`) in the first real cycle.
- [ ] Review the control-vector family and --spec-replace (see ROADMAP.md
      "Next").

## Pending owner smokes

- [ ] Launch a real profile from this branch and confirm the UI text
      renders unchanged (the glyph escapes are byte-identical at runtime,
      the suite covers the strings, but the GUI is the proof).
- [ ] Open a fresh session in this repo and confirm the session-start
      summary prints this file's current phase.
- [ ] Live multi-node test on a GPU worker when one is available.

## Done this cycle

- AGENTS.md, two-line CLAUDE.md, SPEC.md, ROADMAP.md, TASKS.md at the
  root, written fresh (no archived documents migrated).
- Guard tests with README.md, CHANGELOG.md and RPC.md allowlisted and the
  doubled-hyphen check off; non-ASCII glyphs in src and tests spelled as
  escapes, non-ASCII comments reworded.
- CI sanity job added ahead of the matrix job with the same allowlist,
  weekly dependabot (uv and actions), PR template, .superpowers ignored.
- pyproject.toml carries the standard's [tool.ruff] block (line-length 88,
  E, F, W, I, UP, B, RUF, E501 ignored, *.md excluded); ci.yml gains the
  lint job pinned at ruff 0.16.5 between sanity and test.
- 236 findings autofixed, 194 files reformatted, 47 fixed by hand: late
  imports in tests moved to the top, lambdas replaced by partials and
  defs, implicit Optional made explicit, Tier is a StrEnum, the emit
  closure in router_preset binds its profile, ImageInfo is imported for
  typing in core/build_outputs.py (it was an undefined name).
- Names the test suite patches on main_window are explicit re-exports
  (`import x as x`), so the unused-import fix leaves them alone; the two
  tests that patched services.rpc through other modules now patch it
  directly, and core/build_outputs.py types image metadata with a local
  Protocol instead of naming a services class.
- AGENTS.md lint row and gate sentence; CHANGELOG entry for the gate.
- README.md, SPEC.md 3.1 and the pyproject.toml comment state the 3.11
  floor (enum.StrEnum, datetime.UTC); requires-python stays 3.12.
- ci.yml carries a top-level `permissions: contents: read`. Every job only
  reads the checkout, so the default GITHUB_TOKEN needed no write scope.
  Closes CodeQL code scanning alert 1 (actions/missing-workflow-permissions,
  raised on main when default setup first ran), which clears once this
  branch lands.
