#!/usr/bin/env bash
# va-make-vulkan-AMD-780.sh — Vulkan build driver for AMD Radeon 780M iGPU
#
# Usage:
#   ./va-make-vulkan-AMD-780.sh              — full clean build  (configure + compile Release)
#   ./va-make-vulkan-AMD-780.sh make         — same as above
#   ./va-make-vulkan-AMD-780.sh build        — incremental build only  (no reconfigure, no clean)
#   ./va-make-vulkan-AMD-780.sh release      — alias for full clean build
#
# Target: AMD Radeon 780M (RDNA3 iGPU, gfx1103)
# GPU shaders: glslc (Vulkan — no ROCm/HIP runtime required)
# CPU code:    AMD AOCC 5.2.0  (Zen4 AVX512/VNNI-optimised tokeniser, sampler)

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────

AOCC_PATH="${AOCC_PATH:-/opt/AMD/aocc-compiler-5.2.0}"

export AOCC_PATH

# AOCC must lead PATH so cmake finds the right clang/clang++ for CPU-side code
export PATH="$AOCC_PATH/bin:$PATH"

# ── constants ────────────────────────────────────────────────────────────────

PRESET="x64-linux-vulkan-AMD-780-release"
BUILD_DIR="build-x64-linux-vulkan-AMD-780-release"
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

check_file "$AOCC_PATH/bin/clang"   "AOCC clang (CPU compiler)"
check_file "$AOCC_PATH/bin/clang++" "AOCC clang++ (CPU compiler)"
check_cmd  cmake                    "cmake"
check_cmd  ninja                    "ninja"
check_cmd  glslc                    "glslc (Vulkan shader compiler)"

# Confirm AOCC is leading (not system clang)
CLANG_VER=$(clang --version 2>/dev/null | head -1)
[[ "$CLANG_VER" == *"AOCC"* ]] || fail "clang on PATH is not AOCC ('$CLANG_VER') — check AOCC_PATH"

GLSLC_VER=$(glslc --version 2>/dev/null | head -1)

printf '  AOCC_PATH  = %s\n'  "$AOCC_PATH"
printf '  CPU clang  = %s\n'  "$CLANG_VER"
printf '  glslc      = %s\n'  "$GLSLC_VER"
printf '  PRESET     = %s\n'  "$PRESET"
printf '  BUILD_DIR  = %s\n'  "$BUILD_DIR"
printf '  NPROC      = %s\n'  "$NPROC"
printf '─────────────────────────────────────────────────────────────────────────\n\n'

# ── build modes ──────────────────────────────────────────────────────────────

MODE="${1:-make}"

configure() {
    printf '── configure ────────────────────────────────────────────────────────────\n'
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
        [[ -d "$BUILD_DIR" ]] || fail "'$BUILD_DIR' does not exist — run './va-make-vulkan-AMD-780.sh make' first"
        compile
        ;;
    *)
        printf 'Usage: %s [make|build|release]\n' "$0" >&2
        exit 1
        ;;
esac
