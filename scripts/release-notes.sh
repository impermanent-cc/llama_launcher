#!/usr/bin/env bash
# Print one version's section of CHANGELOG.md, for gh's --notes-file. Keeps the
# published release notes and the changelog from drifting -- edit the changelog,
# then cut the release from it:
#
#   ./scripts/release-notes.sh 0.1.0                    # preview on stdout
#   gh release create v0.1.0 --notes-file <(./scripts/release-notes.sh 0.1.0)
#   gh release edit   v0.1.0 --notes-file <(./scripts/release-notes.sh 0.1.0)
#
# A leading "v" is optional, so `v0.1.0` and `0.1.0` both work. Unknown or empty
# sections exit non-zero, so a release can't be cut with blank notes.
set -euo pipefail

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
  echo "usage: $(basename "$0") <version>   (e.g. 0.1.0, or Unreleased)" >&2
  exit 2
fi
VERSION="${VERSION#v}"

# Overridable so the tests can point at a fixture instead of the real file.
CHANGELOG="${CHANGELOG:-$(cd "$(dirname "$0")/.." && pwd)/CHANGELOG.md}"
if [[ ! -f "$CHANGELOG" ]]; then
  echo "no changelog at $CHANGELOG" >&2
  exit 1
fi

# Everything between this version's "## [x.y.z] - date" heading and whatever
# ends it: the next "## " heading, or the link-reference block at the bottom.
notes="$(awk -v ver="$VERSION" '
  /^## / {
    if (inside) exit
    h = $0
    sub(/^##[[:space:]]+/, "", h)
    if (match(h, /^\[[^]]+\]/)) v = substr(h, RSTART + 1, RLENGTH - 2)
    else { split(h, a, /[[:space:]]/); v = a[1] }
    if (v == ver) inside = 1
    next
  }
  inside && /^\[[^]]+\]:[[:space:]]/ { exit }
  inside { print }
' "$CHANGELOG")"

# Drop "### Added"-style subheadings that never got filled in, so a section
# still carrying part of the skeleton doesn't publish bare headings.
notes="$(printf '%s\n' "$notes" | awk '
  function flush() { if (has) printf "%s", buf; buf = ""; has = 0 }
  /^### / { flush(); buf = $0 "\n"; insub = 1; next }
  insub   { buf = buf $0 "\n"; if ($0 ~ /[^[:space:]]/) has = 1; next }
          { print }
  END     { flush() }
')"

# Drop leading blank lines; command substitution already ate the trailing ones.
notes="$(printf '%s\n' "$notes" | sed -e '/./,$!d')"

if [[ -z "$notes" ]]; then
  echo "no notes for version '$VERSION' in $CHANGELOG" >&2
  exit 1
fi

printf '%s\n' "$notes"
