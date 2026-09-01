#!/usr/bin/env bash
# Recapture the upstream llama-server flag list used by
# tests/core/test_catalog_upstream_flags.py.
#
#   ./tests/fixtures/regen_flags.sh b10711
#
# Pulls the small CPU-only server image for that build, so it costs about
# 860 MB rather than the multi-GB `full` image.
set -euo pipefail

# Byte-stable sorting: the default locale collates punctuation differently, so
# a regeneration on another machine produced a 136-line reordering diff with an
# identical flag set. Pin the collation so a real fixture diff always means the
# engine changed.
export LC_ALL=C
BUILD="${1:?usage: regen_flags.sh <build, e.g. b10711>}"
IMAGE="ghcr.io/ggml-org/llama.cpp:server-${BUILD}"
OUT="$(cd "$(dirname "$0")" && pwd)/llama_server_flags_${BUILD}.txt"

podman pull "$IMAGE"
{
  echo "# Flags accepted by llama-server, captured from build ${BUILD}:"
  echo "#   podman run --rm --entrypoint /app/llama-server \\"
  echo "#     ${IMAGE} --help"
  echo "# Regenerate with tests/fixtures/regen_flags.sh when moving to a newer build."
  podman run --rm --entrypoint /app/llama-server "$IMAGE" --help \
    | grep -E '^[[:space:]]*-' \
    | grep -oE '(^|[^A-Za-z0-9-])--?[A-Za-z][A-Za-z0-9-]*' \
    | grep -oE '\-\-?[A-Za-z][A-Za-z0-9-]*' \
    | sort -u
} > "$OUT"
echo "wrote $OUT"
