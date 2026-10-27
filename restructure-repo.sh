#!/usr/bin/env bash
# Restructure vai-llama-cpp-turboquant in place:
#   1. Add/fetch the ggml-org/llama.cpp upstream remote (fork tracking, not submodules
#      -- this repo edits src/ and ggml/ directly, so the dependency can't be a submodule).
#   2. Remove stale build/staging/.deb/.venv bloat from disk (already .gitignore'd, never
#      tracked -- confirmed via `git ls-files` before writing this script).
#   3. Add a host-specific CMake preset for "beast" (RTX 5090 Max-Q, sm_120) to sit next to
#      the existing "borlangy" presets (RTX 4060 sm_89, AMD 780M gfx1103) already in the repo.
#
# Usage:
#   ./restructure-repo.sh            # dry run: shows what would change, deletes nothing
#   ./restructure-repo.sh --apply    # performs the remote add/fetch, deletion, and preset add

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

say()  { printf '\n==> %s\n' "$*"; }
run()  { if [[ $APPLY -eq 1 ]]; then eval "$1"; else printf '[dry-run] %s\n' "$1"; fi; }

if [[ $APPLY -eq 1 ]] && [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree has uncommitted changes. Commit or stash before running with --apply." >&2
  echo "(dry-run without --apply is safe to run regardless)" >&2
  exit 1
fi

# --- 1. upstream remote -----------------------------------------------------
say "Upstream remote"
if git remote get-url upstream >/dev/null 2>&1; then
  echo "upstream already set: $(git remote get-url upstream)"
else
  run "git remote add upstream https://github.com/ggml-org/llama.cpp.git"
fi
run "git fetch upstream --tags"
echo "To sync later: git merge upstream/master   (or rebase, your call per PR)"

# --- 2. delete stale build/staging/.deb/.venv bloat -------------------------
say "Bloat on disk (already gitignored, never tracked -- safe to delete)"
targets=()
for d in build-x64-linux-cpu-aocc-release \
         build-x64-linux-cuda-nvidia-RTX4060-release \
         build-x64-linux-gcc-debug \
         build-x64-linux-hip-rocm10-gfx1103-release \
         build-x64-linux-vulkan-AMD-780-release \
         staging-cpu staging-cuda staging-rocm staging-vulkan \
         .venv; do
  [[ -e "$d" ]] && targets+=("$d")
done
for f in vai-llama.cpp-*.deb; do
  [[ -e "$f" ]] && targets+=("$f")
done

if [[ ${#targets[@]} -eq 0 ]]; then
  echo "nothing to delete"
else
  du -sh "${targets[@]}"
  echo
  echo "DELETION LIST (${#targets[@]} items, listed above with sizes):"
  printf '  %s\n' "${targets[@]}"
  if [[ $APPLY -eq 1 ]]; then
    read -r -p "Delete the items listed above? [y/N] " ans
    if [[ "$ans" == "y" || "$ans" == "Y" ]]; then
      rm -rf -- "${targets[@]}"
      echo "deleted."
    else
      echo "skipped deletion."
    fi
  else
    echo "[dry-run] would prompt to delete the above"
  fi
fi

# --- 3. CMake preset for beast (RTX 5090 Max-Q, sm_120) ---------------------
say "CMake preset for beast (RTX 5090)"
TOOLCHAIN=cmake/x64-linux-cuda-nvidia-rtx5090.cmake
if [[ -f "$TOOLCHAIN" ]]; then
  echo "$TOOLCHAIN already exists, leaving it alone"
else
  if [[ $APPLY -eq 1 ]]; then
    cat > "$TOOLCHAIN" <<'EOF'
# NVIDIA RTX 5090 Max-Q Mobile (Blackwell, sm_120) -- CUDA backend
#
# GPU: NVIDIA CUDA, compiled by nvcc from CUDA toolkit.
# CPU: AMD AOCC (Zen5 AVX512/VNNI-optimised tokeniser, sampler), used both as
# the plain C/C++ compiler and as nvcc's host compiler (-ccbin).
#
# Override paths via env vars:
#   CUDA_PATH   (default: /usr/local/cuda)
#   AOCC_PATH   (default: /opt/AMD/aocc-compiler-5.2.0)

if(DEFINED ENV{CUDA_PATH})
    set(_cuda "$ENV{CUDA_PATH}")
else()
    set(_cuda "/usr/local/cuda")
endif()

if(DEFINED ENV{AOCC_PATH})
    set(_aocc "$ENV{AOCC_PATH}")
else()
    set(_aocc "/opt/AMD/aocc-compiler-5.2.0")
endif()

set(CMAKE_CUDA_COMPILER      "${_cuda}/bin/nvcc"  CACHE FILEPATH "CUDA compiler (nvcc)")
set(CMAKE_C_COMPILER         "${_aocc}/bin/clang"   CACHE FILEPATH "C compiler (AMD AOCC)")
set(CMAKE_CXX_COMPILER       "${_aocc}/bin/clang++" CACHE FILEPATH "C++ compiler (AMD AOCC)")
set(CMAKE_CUDA_HOST_COMPILER "${_aocc}/bin/clang++" CACHE FILEPATH "CUDA host compiler (AMD AOCC)")

set(GGML_CUDA        ON  CACHE BOOL "" FORCE)
set(GGML_HIP         OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN      OFF CACHE BOOL "" FORCE)
set(GGML_METAL       OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE      ON  CACHE BOOL "x86_64 native instructions (Zen5 AVX512/VNNI)" FORCE)

# Blackwell consumer = sm_120. No fallback arch listed: this toolchain file is
# for the RTX 5090 specifically: adjust CMAKE_CUDA_ARCHITECTURES if reused.
set(CMAKE_CUDA_ARCHITECTURES "120" CACHE STRING "CUDA architectures" FORCE)

set(GGML_CUDA_FA_ALL_QUANTS ON CACHE BOOL "" FORCE)

set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
EOF
    echo "wrote $TOOLCHAIN"
  else
    echo "[dry-run] would write $TOOLCHAIN"
  fi
fi

PRESETS=CMakePresets.json
if grep -q '"cuda-nvidia-rtx5090"' "$PRESETS" 2>/dev/null; then
  echo "preset already present in $PRESETS"
else
  if [[ $APPLY -eq 1 ]]; then
    python3 -I - "$PRESETS" <<'PYEOF'
import sys
path = sys.argv[1]
text = open(path).read()
marker = '{ "name": "x64-linux-cuda-nvidia-RTX4060-debug", "inherits": [ "base", "cuda-nvidia-rtx4060", "debug"   ] },'
addition = '''
    {
        "name": "cuda-nvidia-rtx5090",
        "hidden": true,
        "cacheVariables": {
            "CMAKE_TOOLCHAIN_FILE":    "${sourceDir}/cmake/x64-linux-cuda-nvidia-rtx5090.cmake",
            "GGML_CUDA":               "ON",
            "GGML_HIP":                "OFF",
            "GGML_VULKAN":             "OFF",
            "GGML_METAL":              "OFF",
            "GGML_NATIVE":             "ON",
            "CMAKE_CUDA_ARCHITECTURES": "120",
            "GGML_CUDA_FA_ALL_QUANTS": "ON"
        }
    },
    { "name": "x64-linux-cuda-nvidia-RTX5090-release", "inherits": [ "base", "cuda-nvidia-rtx5090", "release" ] },
    { "name": "x64-linux-cuda-nvidia-RTX5090-debug",   "inherits": [ "base", "cuda-nvidia-rtx5090", "debug"   ] },'''
if marker not in text:
    sys.exit("marker not found, refusing to edit " + path)
text = text.replace(marker, marker + addition, 1)
open(path, "w").write(text)
print("inserted RTX5090 presets into " + path)
PYEOF
  else
    echo "[dry-run] would insert cuda-nvidia-rtx5090 preset into $PRESETS"
  fi
fi

say "Done. On beast:   cmake --preset x64-linux-cuda-nvidia-RTX5090-release && cmake --build build-x64-linux-cuda-nvidia-RTX5090-release -j\$(nproc)"
say "     On borlangy: existing presets (x64-linux-cuda-nvidia-RTX4060-release / x64-linux-vulkan-AMD-780-release) are unchanged."
