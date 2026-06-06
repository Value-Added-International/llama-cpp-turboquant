#!/usr/bin/env bash
# va-make.sh — ROCm/HIP gfx1103 build driver with AOCC Zen4 CPU optimizations
#
# Usage:
#   ./va-make.sh              — full clean build  (configure + compile Release)
#   ./va-make.sh make         — same as above
#   ./va-make.sh build        — incremental build only  (no reconfigure, no clean)
#   ./va-make.sh release      — alias for full clean build
#
# Target: AMD Radeon 780M (gfx1103 RDNA3)
# GPU kernels: ROCm 7.13 amdclang  (HIP / rocWMMA FA)
# CPU code:    AMD AOCC 5.2.0      (Zen4 AVX512/VNNI-optimised tokeniser, sampler)

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────

ROCM_PATH="${ROCM_PATH:-/opt/rocm/core-7.13}"
AOCC_PATH="${AOCC_PATH:-/opt/AMD/aocc-compiler-5.2.0}"
ROCWMMA_PATH="${ROCWMMA_PATH:-/opt/rocwmma-rocm7.13}"

export ROCM_PATH
export HIP_PATH="$ROCM_PATH"
export HIPCXX="$ROCM_PATH/lib/llvm/bin/clang"
export GPU_TARGETS="gfx1103"
export ROCWMMA_PATH

# AOCC must lead PATH so cmake finds the right clang/clang++ for CPU-side code
export PATH="$AOCC_PATH/bin:$PATH"

# ── constants ────────────────────────────────────────────────────────────────

PRESET="x64-linux-hip-rocm713-gfx1103-release"
BUILD_DIR="build-x64-linux-hip-rocm713-gfx1103-release"
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

check_file "$ROCM_PATH/lib/llvm/bin/clang"            "ROCm amdclang (GPU compiler)"
check_file "$ROCM_PATH/include/hip/hip_runtime.h"     "HIP headers"
check_file "$ROCWMMA_PATH/include/rocwmma/rocwmma-version.hpp" "rocWMMA headers"
check_file "$AOCC_PATH/bin/clang"                     "AOCC clang (CPU compiler)"
check_file "$AOCC_PATH/bin/clang++"                   "AOCC clang++ (CPU compiler)"
check_cmd  cmake                                      "cmake"
check_cmd  ninja                                      "ninja"

# Confirm AOCC is leading (not system clang)
CLANG_VER=$(clang --version 2>/dev/null | head -1)
[[ "$CLANG_VER" == *"AOCC"* ]] || fail "clang on PATH is not AOCC ('$CLANG_VER') — check AOCC_PATH"

printf '  ROCM_PATH    = %s\n'  "$ROCM_PATH"
printf '  HIP_PATH     = %s\n'  "$HIP_PATH"
printf '  HIPCXX       = %s\n'  "$HIPCXX"
printf '  ROCWMMA_PATH = %s\n'  "$ROCWMMA_PATH"
printf '  GPU_TARGETS  = %s\n'  "$GPU_TARGETS"
printf '  CPU clang    = %s\n'  "$CLANG_VER"
printf '  AOCC_PATH    = %s\n'  "$AOCC_PATH"
printf '  PRESET     = %s\n'  "$PRESET"
printf '  BUILD_DIR  = %s\n'  "$BUILD_DIR"
printf '  NPROC      = %s\n'  "$NPROC"
printf '─────────────────────────────────────────────────────────────────────────\n\n'

# ── build modes ──────────────────────────────────────────────────────────────

MODE="${1:-make}"

configure() {
    printf '── configure ────────────────────────────────────────────────────────────\n'
    cmake --preset "$PRESET"
    # ROCm amdclang does not auto-mkdir for -MF depfile paths. cmake generates
    # the parent ggml-hip.dir/__/ggml-cuda/ but misses the template-instances
    # subdirectory. All fattn-vec-instance-*.cu jobs fail immediately at -j>1
    # if it doesn't exist before the parallel build starts.
    mkdir -p "$BUILD_DIR/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/template-instances"
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
        [[ -d "$BUILD_DIR" ]] || fail "'$BUILD_DIR' does not exist — run './va-make.sh make' first"
        compile
        ;;
    *)
        printf 'Usage: %s [make|build|release]\n' "$0" >&2
        exit 1
        ;;
esac
