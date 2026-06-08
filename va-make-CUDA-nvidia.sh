#!/usr/bin/env bash
# va-make-CUDA-nvidia.sh — NVIDIA CUDA RTX 4060 build driver
#
# Usage:
#   ./va-make-CUDA-nvidia.sh              — full clean build  (configure + compile Release)
#   ./va-make-CUDA-nvidia.sh make         — same as above
#   ./va-make-CUDA-nvidia.sh build        — incremental build only  (no reconfigure, no clean)
#   ./va-make-CUDA-nvidia.sh release      — alias for full clean build
#
# Target: NVIDIA GeForce RTX 4060 Laptop GPU (Ada Lovelace, sm_89, 8 GiB GDDR6)
# GPU kernels: CUDA 13.3 nvcc  (sm_89 — Ada Lovelace; sm_86 fat binary also built)
# CPU code:    system GCC/clang (AOCC is AMD-only; CUDA builds use the system toolchain)

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────

CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-13.3}"

export CUDA_PATH

# ── constants ────────────────────────────────────────────────────────────────

PRESET="x64-linux-cuda-nvidia-RTX4060-release"
BUILD_DIR="build-x64-linux-cuda-nvidia-RTX4060-release"
NPROC=$(nproc)

# ── preflight checks ─────────────────────────────────────────────────────────

fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

check_file() {
    [[ -f "$1" ]] || fail "$2 not found at: $1"
}

check_cmd() {
    command -v "$1" >/dev/null 2>&1 || fail "$2 not found — is it installed and on PATH?"
}

printf '\n── environment check ────────────────────────────────────────────────────\n'

check_file "$CUDA_PATH/bin/nvcc"                "nvcc (CUDA compiler)"
check_file "$CUDA_PATH/include/cuda_runtime.h"  "CUDA runtime headers"
check_cmd  cmake                                "cmake"
check_cmd  ninja                                "ninja"

NVCC_VER=$("$CUDA_PATH/bin/nvcc" --version 2>/dev/null | grep "release" | head -1)
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "nvidia-smi unavailable")

printf '  CUDA_PATH  = %s\n'  "$CUDA_PATH"
printf '  nvcc       = %s\n'  "$NVCC_VER"
printf '  GPU        = %s\n'  "$GPU_INFO"
printf '  PRESET     = %s\n'  "$PRESET"
printf '  BUILD_DIR  = %s\n'  "$BUILD_DIR"
printf '  NPROC      = %s\n'  "$NPROC"
printf '─────────────────────────────────────────────────────────────────────────\n\n'

# ── build modes ──────────────────────────────────────────────────────────────

MODE="${1:-make}"

configure() {
    printf '── configure ────────────────────────────────────────────────────────────\n'
    # Make nvcc visible to CMake's find_package(CUDAToolkit)
    export PATH="$CUDA_PATH/bin:$PATH"
    cmake --preset "$PRESET"
}

compile() {
    printf '\n── build (Release, -j%s) ─────────────────────────────────────────────────\n' "$NPROC"
    cmake --build "$BUILD_DIR" -j"$NPROC"
    printf '\n── build complete ───────────────────────────────────────────────────────\n'
    printf '  binaries: %s/bin/\n\n' "$BUILD_DIR"
}

case "$MODE" in
    make|release)
        rm -rf "$BUILD_DIR"
        configure
        compile
        ;;
    build)
        [[ -d "$BUILD_DIR" ]] || fail "'$BUILD_DIR' does not exist — run './va-make-CUDA-nvidia.sh make' first"
        export PATH="$CUDA_PATH/bin:$PATH"
        compile
        ;;
    *)
        printf 'Usage: %s [make|build|release]\n' "$0" >&2
        exit 1
        ;;
esac
