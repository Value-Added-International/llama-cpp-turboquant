#!/usr/bin/env bash
#
# Build the self-contained vai-llama.cpp-NVIDIA .deb package.
#
# Reuses the already-verified x64-linux-cuda-nvidia-RTX4060-release preset
# (AOCC 5.2.0 as both the plain C/C++ compiler and nvcc's host compiler,
# CUDA 13.3 nvcc for device code, sm_89;86) rather than a separate hand-rolled
# cmake flag set, so the packaged binaries are built exactly the same way as
# the validated dev build.
#
# Usage:
#   ./.devops/build-cuda-package.sh
#
# Must be run from the repository root (same convention as the other
# va-make-*.sh build drivers).

set -euo pipefail

# A stale LD_LIBRARY_PATH inherited from the invoking shell (e.g. leftover
# ROCm dev-environment exports) takes priority over RUNPATH in glibc's
# search order and can silently corrupt library resolution / verification
# below. Build and verify against a clean environment.
unset LD_LIBRARY_PATH

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PRESET="x64-linux-cuda-nvidia-RTX4060-release"
BUILD_DIR="build-${PRESET}"
STAGING_DIR="staging-cuda"
INSTALL_PREFIX="/opt/vai-llama/nvidia"
PACKAGE_NAME="vai-llama.cpp-NVIDIA"
VERSION="0.0.$(git rev-list --count HEAD)"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-13.3}"
AOCC_PATH="${AOCC_PATH:-/opt/AMD/aocc-compiler-5.2.0}"

fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

[[ -d "$BUILD_DIR" ]] || fail "'$BUILD_DIR' does not exist — run ./va-make-CUDA-nvidia.sh make first"

echo "=== $PACKAGE_NAME package builder ==="
echo "Version: $VERSION"
echo "Install prefix: $INSTALL_PREFIX"
echo ""

# Step 1: install the already-built binaries into the staging tree
echo "[1/4] Installing $PRESET build into staging..."
rm -rf "$STAGING_DIR"
cmake --install "$BUILD_DIR" --prefix "$STAGING_DIR$INSTALL_PREFIX"

# Step 2: bundle CUDA + system runtime libraries into lib/ (see the Vulkan
# script for why libs land in lib/, not alongside binaries in bin/).
# libcuda.so (the NVIDIA driver interface) is deliberately NOT bundled --
# it's version-locked to the host's installed NVIDIA driver and must come
# from the host, not the package.
echo "[2/4] Bundling CUDA + system runtime libraries..."
LIB_DIR="$STAGING_DIR$INSTALL_PREFIX/lib"
mkdir -p "$LIB_DIR"

bundle_lib() {
    local pattern="$1"
    local found
    found=$(find "$LIB_DIR" -maxdepth 1 -name "$pattern" -type f 2>/dev/null | head -1)
    if [[ -n "$found" ]]; then
        return 0
    fi
    for path in "$CUDA_PATH/lib64" "$AOCC_PATH/lib" /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib; do
        [[ -d "$path" ]] || continue
        # Match both the real file AND its SONAME symlink (e.g.
        # libstdc++.so.6 -> libstdc++.so.6.0.35) -- the loader resolves by
        # exact SONAME, and `-type f` alone silently skips the symlink,
        # leaving the binary unable to find the bundled copy at runtime.
        local matches
        matches=$(find -L "$path" -maxdepth 1 -name "$pattern" \( -type f -o -type l \) 2>/dev/null)
        if [[ -n "$matches" ]]; then
            echo "  bundling $pattern from $path"
            while IFS= read -r m; do
                cp -P "$m" "$LIB_DIR/"
                # Raw system/CUDA libraries carry no $ORIGIN-relative RPATH
                # of their own, so their own dependencies (e.g. libcurl ->
                # libz) still resolve from the system instead of our bundled
                # copies. Force $ORIGIN so cross-references stay bundled.
                if [[ -f "$m" && ! -L "$m" ]]; then
                    patchelf --set-rpath '$ORIGIN' "$LIB_DIR/$(basename "$m")" 2>/dev/null || true
                fi
            done <<< "$matches"
            return 0
        fi
    done
    fail "$pattern not found anywhere searched -- package would be non-self-contained"
}
for lib in "libcudart.so*" "libcublas.so*" "libcublasLt.so*" "libnvrtc.so*" "libnvrtc-builtins.so*"; do
    bundle_lib "$lib"
done
# libomp.so (AOCC's LLVM OpenMP runtime, needed by libggml-cpu -- not
# libgomp, GNU's runtime, which this AOCC-compiled project does not use)
for lib in "libomp.so*" "libstdc++.so.*" "libgcc_s.so.*" "libssl.so*" "libcrypto.so*" "libcurl.so*" "libz.so*" "libzstd.so*"; do
    bundle_lib "$lib"
done

# Step 3: write the DEBIAN control metadata and build the package
echo "[3/4] Writing package metadata..."
mkdir -p "$STAGING_DIR/DEBIAN"
cat > "$STAGING_DIR/DEBIAN/control" << EOF
Package: $PACKAGE_NAME
Version: $VERSION
Architecture: amd64
Maintainer: Value-Added International <value.added.kr@gmail.com>
Section: science
Priority: optional
Description: llama.cpp + TurboQuant CUDA backend (RTX 4060 / AOCC Zen4)
 Self-contained CUDA-accelerated build of llama-cpp-turboquant, compiled with
 AMD AOCC 5.2.0 as both the CPU compiler and nvcc's host compiler, targeting
 NVIDIA Ada Lovelace (sm_89) and Ampere (sm_86). Includes MXFP4/NVFP4 quant
 type support (NVFP4 runs via the CUDA fallback path on non-Blackwell GPUs).
 Bundles the CUDA runtime and required system libraries.
 .
 Requirements (host system): an NVIDIA GPU with a driver providing libcuda.so
 compatible with the bundled CUDA 13.3 runtime. libcuda.so itself is NOT
 bundled -- it is version-locked to the host driver by design.
 .
 Installation location: $INSTALL_PREFIX
EOF
cat > "$STAGING_DIR/DEBIAN/postinst" << 'EOF'
#!/bin/sh
exit 0
EOF
chmod 755 "$STAGING_DIR/DEBIAN/postinst"

DEB_FILE="${PACKAGE_NAME}_${VERSION}_amd64.deb"
dpkg-deb --build --root-owner-group "$STAGING_DIR" "$DEB_FILE"

# Step 4: verify
# A plain "not found" grep only catches libraries that fail to resolve
# anywhere -- it would NOT have caught libomp.so silently resolving via this
# dev machine's system-wide AOCC ldconfig registration instead of the
# bundled copy (a real bug found while building the Vulkan package). Check
# that every non-glibc, non-driver dependency resolves from OUR lib/.
echo "[4/4] Verifying package..."
dpkg-deb -I "$DEB_FILE" > /dev/null && echo "  package structure OK"

# libcuda.so / libnvidia-ml.so are the NVIDIA driver interface -- version-
# locked to the host driver by design, deliberately not bundled.
LIB_DIR_REAL=$(realpath -m "$LIB_DIR")
ALLOWED_HOST_LIBS='^(linux-vdso\.so|ld-linux-x86-64\.so|libc\.so|libm\.so|libpthread\.so|libdl\.so|librt\.so|libcuda\.so|libnvidia-ml\.so)'
NOT_SELF_CONTAINED=0
while IFS= read -r line; do
    [[ "$line" == *"=>"* ]] || continue
    libname=$(awk '{print $1}' <<< "$line")
    libpath=$(awk '{print $3}' <<< "$line")
    [[ "$libname" =~ $ALLOWED_HOST_LIBS ]] && continue
    if [[ -z "$libpath" || "$libpath" == "not" ]]; then
        echo "  MISSING: $libname"
        NOT_SELF_CONTAINED=1
        continue
    fi
    resolved=$(realpath -m "$libpath")
    if [[ "$resolved" != "$LIB_DIR_REAL"/* ]]; then
        echo "  NOT BUNDLED: $libname resolved from $libpath (expected under $LIB_DIR)"
        NOT_SELF_CONTAINED=1
    fi
done < <(ldd "$STAGING_DIR$INSTALL_PREFIX/bin/llama-cli" 2>&1)
[[ $NOT_SELF_CONTAINED -eq 0 ]] && echo "  all libraries resolve from the bundled lib/ (or are glibc/NVIDIA-driver primitives)" || fail "package is not self-contained -- see above"

echo ""
echo "=== Build complete: $DEB_FILE ($(du -h "$DEB_FILE" | cut -f1)) ==="
echo "Install: sudo dpkg -i $DEB_FILE"
echo "Binary after install: $INSTALL_PREFIX/bin/llama-cli"
