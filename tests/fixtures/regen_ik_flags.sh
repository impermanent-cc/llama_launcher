#!/usr/bin/env bash
# Recapture the ik_llama.cpp flag list used by
# tests/core/test_catalog_upstream_flags.py, and re-probe which shared settings
# ik actually rejects (the source for MAINLINE_ONLY_FLAGS).
#
#   ./tests/fixtures/regen_ik_flags.sh
#
# Why this is not just a --help dump like the mainline script: ik's help
# UNDER-REPORTS. --n-gpu-layers, --n-predict, --embeddings and --alias are all
# accepted by its parser while absent from --help, so trusting the help text
# alone wrongly marks them unsupported. Every candidate is therefore executed
# and checked for "unknown argument".
#
# Note the image is CUDA-only and its binary needs libcuda.so.1 present just to
# start, even for --help. On a host without an NVIDIA runtime, build a stub:
#
#   printf 'void cuInit(void){}\n' > /tmp/stub.c
#   gcc -shared -fPIC -o /tmp/libcuda.so.1 /tmp/stub.c
#   export IK_STUB=/tmp/libcuda.so.1
#
# A CPU-only ik image is NOT a valid substitute: it omits the GPU flags
# entirely, which would misreport them as unsupported.
set -euo pipefail

IMAGE="${IK_IMAGE:-ghcr.io/ikawrakow/ik-llama-cpp:cu12-server}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/ik_llama_server_flags_cu12.txt"

mount_args=()
if [[ -n "${IK_STUB:-}" ]]; then
  mount_args=(-v "${IK_STUB}:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro")
fi

run_ik() { podman run --rm "${mount_args[@]}" "$IMAGE" "$@" 2>&1; }

podman pull "$IMAGE"

# 1. Everything --help prints.
help_flags="$(run_ik --help \
  | grep -E '^[[:space:]]*-' \
  | grep -oE '(^|[^A-Za-z0-9-])--?[A-Za-z][A-Za-z0-9-]*' \
  | grep -oE '\-\-?[A-Za-z][A-Za-z0-9-]*' \
  | sort -u || true)"

# 2. Undocumented-but-accepted extras, confirmed by execution.
extras=""
for f in --n-gpu-layers --n-predict --embeddings --alias; do
  if ! run_ik "$f" 0 -m /nonexistent.gguf | grep -q 'unknown argument'; then
    extras+="$f"$'\n'
  fi
done

{
  echo "# Flags accepted by ik_llama.cpp's llama-server, from build cu12-server."
  echo "#"
  echo "# TWO sources, because ik's --help under-reports what its parser accepts:"
  echo "#   1. every flag printed by 'llama-server --help'"
  echo "#   2. flags confirmed accepted by EXECUTING them (undocumented but valid):"
  echo "#      $(echo "$extras" | tr '\n' ' ' | sed 's/ *$//')"
  echo "#"
  echo "# Regenerate with tests/fixtures/regen_ik_flags.sh."
  printf '%s\n%s' "$help_flags" "$extras" | grep -E '^-' | sort -u
} > "$OUT"
echo "wrote $OUT"
