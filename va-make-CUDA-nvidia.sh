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
# CPU code:    AMD AOCC 5.2.0 (Zen4 AVX512/VNNI-optimised tokeniser, sampler),
#              also used as nvcc's host compiler (-ccbin) -- verified working.

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────

CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-13.4}"
AOCC_PATH="${AOCC_PATH:-/opt/AMD/aocc-compiler-5.2.0}"

export CUDA_PATH
export AOCC_PATH

# AOCC must lead PATH so cmake finds the right clang/clang++ for CPU-side code
export PATH="$AOCC_PATH/bin:$PATH"

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
check_file "$AOCC_PATH/bin/clang"               "AOCC clang (CPU/host compiler)"
check_file "$AOCC_PATH/bin/clang++"             "AOCC clang++ (CPU/host compiler)"
check_cmd  cmake                                "cmake"
check_cmd  ninja                                "ninja"

# Confirm AOCC is leading (not system clang)
# NB: capture via command substitution (reads to EOF) then slice the first
# line in pure bash -- piping straight into `head -1` can SIGPIPE the live
# process if head closes early, which AOCC's multi-line `clang --version`
# output triggers intermittently under `set -o pipefail`.
CLANG_VER_FULL=$(clang --version 2>/dev/null || true)
CLANG_VER=${CLANG_VER_FULL%%$'\n'*}
[[ "$CLANG_VER" == *"AOCC"* ]] || fail "clang on PATH is not AOCC ('$CLANG_VER') — check AOCC_PATH"

# NB: capture via command substitution (reads to EOF) before filtering/slicing
# in pure bash -- piping a live process straight into `grep`/`head` can
# SIGPIPE it if the downstream reader closes early on a multi-line write.
NVCC_FULL=$("$CUDA_PATH/bin/nvcc" --version 2>/dev/null || true)
NVCC_VER=""
while IFS= read -r _line; do
    [[ "$_line" == *release* ]] && { NVCC_VER="$_line"; break; }
done <<< "$NVCC_FULL"
GPU_INFO_FULL=$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable")
GPU_INFO=${GPU_INFO_FULL%%$'\n'*}

printf '  CUDA_PATH  = %s\n'  "$CUDA_PATH"
printf '  nvcc       = %s\n'  "$NVCC_VER"
printf '  AOCC_PATH  = %s\n'  "$AOCC_PATH"
printf '  CPU clang  = %s\n'  "$CLANG_VER"
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
