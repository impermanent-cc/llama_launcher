# AI assistance disclosure

Llama Launcher's codebase is developed with AI assistance under the project
owner's direction. Every change is human-reviewed, and the owner is responsible
for it before release.

## Tools and models

- Claude Code (Anthropic's CLI coding agent, including its subagent
  orchestration) is the primary tool used to develop this project.
- The Anthropic Claude model family provides the underlying models.
- Per-change model detail lives in the project's own session records, not in the
  git history.

## Attribution policy

This project uses a single repository-level disclosure of AI assistance, this
file plus the short pointer in the README, instead of per-commit attribution
trailers. Credit is presented cohesively here rather than scattered across
individual commit messages.

This policy takes effect for commits made on or after 2026-08-24. Some earlier
commits carry a `Co-Authored-By` trailer from the project's previous practice;
those are left unchanged.

## Responsibility

The human owner reviews, tests, and certifies every release. AI output is never
merged sight-unseen, and the AI does not self-certify its own work.
