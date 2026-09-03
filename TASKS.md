# llama_launcher: tasks

## Current phase

Idle: no cycle open. The workflow standard's stamp and its ruff gate
landed on main 2026-09-03 as merge f37f6a5, a --no-ff merge of
chore/ruff-gate with chore/workflow-stamp beneath it. Both branches are
gone, locally and on origin, and origin now carries main alone.

actions/checkout is at v7 and actions/setup-python at v7, the two bumps
dependabot opened as PRs 2 and 3 once the config landed; both PRs are
closed against the commit that applied them and their branches deleted.
Neither project breaking change reaches this workflow: it has no
pull_request_target or workflow_run trigger, and it never passed
setup-python's removed pip-install input.

0.1.1 is cut in CHANGELOG.md and pyproject.toml but is not tagged, so the
changelog's 0.1.1 link and the AGENTS.md "released (v0.1.0)" line resolve
only once the tag exists. To finish the release:

    git tag -a v0.1.1 -m "llama_launcher v0.1.1"
    git push origin v0.1.1
    gh release create v0.1.1 --notes-file <(./scripts/release-notes.sh 0.1.1)

then update the AGENTS.md version line.

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
- [ ] Eight settings-catalog group names contain "&" ("Model & Context",
      "GPU & Memory", "CPU & Threading", "Embedding & Reranking",
      "Multi-GPU & Graph", "Networking & CORS", "Performance & Batching",
      "Server & Tools"). configure_panel builds each QGroupBox with the
      name verbatim, and Qt reads "&" as a mnemonic, so the title renders
      with an underlined space: "Model_Context". Predates this cycle (the
      strings are unchanged since the catalog commits). Fix by doubling
      the ampersand for display, not in the catalog data, since the group
      name is also a lookup key.
- [ ] `--dry-run` prints a konsole invocation on a box that has no
      konsole. app.py takes `load_config(base_dir).get("terminal",
      DEFAULT_TEMPLATE)` and DEFAULT_TEMPLATE is TERMINALS[0], konsole,
      so with no "terminal" key in config.json the preview never consults
      detect_terminal(). The launch path is correct: terminal.launch()
      with template=None detects, and stripping konsole from PATH walks
      gnome-terminal, xfce4-terminal, xterm as intended. So the preview
      and the launch disagree wherever konsole is absent. Dates to
      285592b, the commit that added --dry-run. Fix by falling back to
      detect_terminal() before DEFAULT_TEMPLATE.

## Pending owner smokes

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
- 0.1.1 cut: the accumulated [Unreleased] entries became
  `## [0.1.1] - 2026-09-03`, a fresh skeleton opened above it, the link
  references gained a 0.1.1 row, and pyproject.toml reads 0.1.1. The bump
  is carried by the two user-visible fixes already on main (the dialog
  title suffix and the Configure form's first-start rows); nothing in the
  stamp or the ruff gate is user-visible.
