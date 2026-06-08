#!/usr/bin/env bash
# va-make-bench-only.sh — incremental build of llama-bench + llama-batched-bench only
#
# Usage:
#   ./va-make-bench-only.sh              — NVIDIA CUDA RTX 4060 (default)
#   ./va-make-bench-only.sh cuda         — NVIDIA CUDA RTX 4060
#   ./va-make-bench-only.sh rocm         — AMD HIP gfx1103
#   ./va-make-bench-only.sh vulkan       — Vulkan (AMD 780M)
#
# Builds only the two bench targets so you don't have to wait for a full recompile.
# Run the relevant va-make-*.sh first to configure + do the initial build.

set -euo pipefail

BACKEND="${1:-cuda}"
NPROC=$(nproc)

fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

case "$BACKEND" in
    cuda|nvidia)
        BUILD_DIR="build-x64-linux-cuda-nvidia-RTX4060-release"
        CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-13.3}"
        export PATH="$CUDA_PATH/bin:$PATH"
        ;;
    rocm|hip)
        BUILD_DIR="build-x64-linux-hip-rocm713-gfx1103-release"
        ;;
    vulkan)
        BUILD_DIR="build-x64-linux-vulkan-AMD-780-release"
        ;;
    *)
        printf 'Usage: %s [cuda|rocm|vulkan]\n' "$0" >&2
        exit 1
        ;;
esac

[[ -d "$BUILD_DIR" ]] || fail "'$BUILD_DIR' does not exist — run the matching va-make-*.sh first"

printf '\n── bench-only build  [%s]  -j%s ────────────────────────────────────────\n' "$BUILD_DIR" "$NPROC"

cmake --build "$BUILD_DIR" -j"$NPROC" \
    --target llama-bench \
    --target llama-batched-bench

printf '\n── done ─────────────────────────────────────────────────────────────────\n'
printf '  %s/bin/llama-bench\n' "$BUILD_DIR"
printf '  %s/bin/llama-batched-bench\n\n' "$BUILD_DIR"
