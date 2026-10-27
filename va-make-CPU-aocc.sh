#!/usr/bin/env bash
# va-make-CPU-aocc.sh — CPU-only build driver for AMD Ryzen 7 8845HS (Zen4)
#
# Usage:
#   ./va-make-CPU-aocc.sh              — full clean build  (configure + compile Release)
#   ./va-make-CPU-aocc.sh make         — same as above
#   ./va-make-CPU-aocc.sh build        — incremental build only  (no reconfigure, no clean)
#   ./va-make-CPU-aocc.sh release      — alias for full clean build
#
# Target: AMD Ryzen 7 8845HS (Zen4, AVX-512/VNNI) — no GPU backend
# CPU code: AMD AOCC 5.2.0 (Zen4 AVX512/VNNI-optimised, native march)

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────

AOCC_PATH="${AOCC_PATH:-/opt/AMD/aocc-compiler-5.2.0}"

export AOCC_PATH

# AOCC must lead PATH so cmake finds the right clang/clang++ for CPU-side code
export PATH="$AOCC_PATH/bin:$PATH"

# ── constants ────────────────────────────────────────────────────────────────

PRESET="x64-linux-cpu-aocc-release"
BUILD_DIR="build-x64-linux-cpu-aocc-release"
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

# Confirm AOCC is leading (not system clang)
# NB: capture via command substitution (reads to EOF) then slice the first
# line in pure bash -- piping straight into `head -1` can SIGPIPE the live
# process if head closes early, which AOCC's multi-line `clang --version`
# output triggers intermittently under `set -o pipefail`.
CLANG_VER_FULL=$(clang --version 2>/dev/null || true)
CLANG_VER=${CLANG_VER_FULL%%$'\n'*}
[[ "$CLANG_VER" == *"AOCC"* ]] || fail "clang on PATH is not AOCC ('$CLANG_VER') — check AOCC_PATH"

printf '  AOCC_PATH  = %s\n'  "$AOCC_PATH"
printf '  CPU clang  = %s\n'  "$CLANG_VER"
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
        [[ -d "$BUILD_DIR" ]] || fail "'$BUILD_DIR' does not exist — run './va-make-CPU-aocc.sh make' first"
        compile
        ;;
    *)
        printf 'Usage: %s [make|build|release]\n' "$0" >&2
        exit 1
        ;;
esac
