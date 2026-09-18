#!/usr/bin/env bash
# Recapture the PrismML llama.cpp flag list used by
# tests/core/test_catalog_upstream_flags.py from a release tarball.
#
#   ./tests/fixtures/regen_prism_flags.sh prism-b10685-7dffb15
#
# The fork publishes no container image, so the capture downloads the
# release's Ubuntu x64 CPU tarball (about 17 MB) and runs its llama-server
# --help. The binary carries an $ORIGIN runpath and finds its own libraries
# without an environment variable. After a regeneration, update PRISM_FIXTURE
# in the test and run the suite: a catalogued flag the release rejects and an
# accepted flag that is neither catalogued nor recorded as unexposed both
# fail, and each has to be decided.
set -euo pipefail

# Byte-stable sorting, so a fixture diff always means the engine changed.
export LC_ALL=C

TAG="${1:?usage: regen_prism_flags.sh <release tag, e.g. prism-b10685-7dffb15>}"
BUILD="$(printf '%s' "$TAG" | cut -d- -f2)"
: "${BUILD:?tag must look like prism-b<N>-<sha>}"
ASSET="llama-${TAG}-bin-ubuntu-x64.tar.gz"
URL="https://github.com/PrismML-Eng/llama.cpp/releases/download/${TAG}/${ASSET}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${HERE}/prism_llama_server_flags_${BUILD}.txt"
WORK="$(mktemp -d /tmp/prism-flags.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

curl -sSfL -o "${WORK}/${ASSET}" "$URL"
tar -xzf "${WORK}/${ASSET}" -C "$WORK"
SERVER="$(find "$WORK" -type f -name llama-server -print -quit)"
chmod +x "$SERVER"
{
  echo "# Flags accepted by PrismML llama.cpp's llama-server, captured from release ${TAG}:"
  echo "#   ${URL}"
  echo "#   llama-server --help"
  echo "# Regenerate with tests/fixtures/regen_prism_flags.sh <tag> when moving to a newer release."
  "$SERVER" --help \
    | grep -E '^[[:space:]]*-' \
    | grep -oE '(^|[^A-Za-z0-9-])--?[A-Za-z][A-Za-z0-9-]*' \
    | grep -oE '\-\-?[A-Za-z][A-Za-z0-9-]*' \
    | sort -u
} > "${WORK}/flags.txt"
mv "${WORK}/flags.txt" "$OUT"
echo "wrote $OUT"
