# llama_launcher: tasks

## Current phase

Branch chore/workflow-stamp, started 2026-09-02: bring the repository to
the workflow standard with the light touch for a live, public project. New
root documents, the guard tests, a CI sanity job ahead of the existing
matrix job, dependabot and a PR template exist; UI glyphs in code are \u
escapes; README, CHANGELOG, RPC.md and packaging are untouched; the suite
is green. Next: the owner reads the diff, merges with --no-ff, deletes the
branch and pushes. Updated 2026-09-02.

## Open items

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
