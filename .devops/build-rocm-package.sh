#!/usr/bin/env bash
#
# Build the self-contained vai-llama.cpp-ROCM .deb package.
#
# Reuses the already-verified x64-linux-hip-rocm10-gfx1103-release preset
# (AOCC 5.2.0 CPU code + ROCm 10.0 amdclang HIP kernels, gfx1103/RDNA3,
# rocWMMA flash attention) rather than a separate hand-rolled cmake flag set.
#
# Unlike the Vulkan/CUDA packages, ROCm's own shared libraries are NOT
# flattened into a single lib/ directory: several of them (libamdhip64,
# librocroller, the LLVM/clang JIT backend librocroller pulls in) carry
# relative $ORIGIN-based RPATHs that expect ROCm's own subdirectory layout
# (lib/rocm_sysdeps/lib/, lib/llvm/lib/). This script discovers the full
# transitive closure via `ldd` and copies each library to the SAME path,
# relative to /opt/rocm/lib, that it occupies on the build host -- so those
# vendor RPATHs keep resolving without needing any RPATH patching.
#
# Usage:
#   ./.devops/build-rocm-package.sh
#
# Must be run from the repository root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PRESET="x64-linux-hip-rocm10-gfx1103-release"
BUILD_DIR="build-${PRESET}"
STAGING_DIR="staging-rocm"
INSTALL_PREFIX="/opt/vai-llama/rocm"
PACKAGE_NAME="vai-llama.cpp-ROCM"
VERSION="0.0.$(git rev-list --count HEAD)"
ROCM_LIB="/opt/rocm/lib"
AOCC_PATH="${AOCC_PATH:-/opt/AMD/aocc-compiler-5.2.0}"

fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

[[ -d "$BUILD_DIR" ]] || fail "'$BUILD_DIR' does not exist — run ./va-make-ROCM.sh make first"
[[ -d "$ROCM_LIB" ]]  || fail "$ROCM_LIB not found — is ROCm installed?"

echo "=== $PACKAGE_NAME package builder ==="
echo "Version: $VERSION"
echo "Install prefix: $INSTALL_PREFIX"
echo ""

# Step 1: install the already-built binaries into the staging tree
echo "[1/4] Installing $PRESET build into staging..."
rm -rf "$STAGING_DIR"
cmake --install "$BUILD_DIR" --prefix "$STAGING_DIR$INSTALL_PREFIX"

# Step 2: discover and bundle the full ROCm runtime closure
echo "[2/4] Discovering + bundling ROCm runtime library closure..."
LIB_DIR="$STAGING_DIR$INSTALL_PREFIX/lib"

# ldd resolves the *entire* transitive dependency tree in one pass as long as
# every library it needs is already reachable via LD_LIBRARY_PATH -- no
# manual recursion needed. Union the closure across every .so we installed,
# since different tools may pull in slightly different subsets.
export LD_LIBRARY_PATH="$ROCM_LIB:${LD_LIBRARY_PATH:-}"

ROCM_CLOSURE=$(
    for f in "$LIB_DIR"/*.so "$LIB_DIR"/*.so.* "$STAGING_DIR$INSTALL_PREFIX/bin/llama-cli"; do
        [[ -f "$f" ]] || continue
        ldd "$f" 2>/dev/null | awk -v rocm="$ROCM_LIB" '$3 ~ "^" rocm {print $3}'
    done | sort -u
)

if [[ -z "$ROCM_CLOSURE" ]]; then
    fail "no ROCm libraries found via ldd -- is LD_LIBRARY_PATH/ROCM_LIB correct?"
fi

while IFS= read -r src; do
    [[ -n "$src" ]] || continue
    rel="${src#"$ROCM_LIB"/}"
    dest="$LIB_DIR/$rel"
    mkdir -p "$(dirname "$dest")"
    echo "  bundling $rel"
    cp -P "$src" "$dest"
    # ldd reports the SONAME symlink (e.g. libamdhip64.so.7), not the real
    # versioned file it points to (libamdhip64.so.7.15.26333-0000000). Since
    # NEEDED entries reference the symlink name, that's what we must bundle
    # -- but the symlink is dangling unless its target is ALSO copied
    # alongside it. ROCm's own symlinks here are all relative (target is a
    # bare filename, same directory), so the target lives beside $src too.
    if [[ -L "$src" ]]; then
        target=$(readlink "$src")
        if [[ "$target" != /* ]]; then
            src_target="$(dirname "$src")/$target"
            dest_target="$(dirname "$dest")/$target"
            if [[ -f "$src_target" && ! -f "$dest_target" ]]; then
                echo "  bundling $(dirname "$rel")/$target (symlink target)"
                cp -P "$src_target" "$dest_target"
            fi
        fi
    fi
done <<< "$ROCM_CLOSURE"

# Bundle non-ROCm system libraries the same way the Vulkan/CUDA scripts do.
bundle_lib() {
    local pattern="$1"
    local found
    found=$(find "$LIB_DIR" -maxdepth 1 -name "$pattern" -type f 2>/dev/null | head -1)
    [[ -n "$found" ]] && return 0
    for path in "$AOCC_PATH/lib" /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib; do
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
                # Raw system libraries carry no $ORIGIN-relative RPATH of
                # their own, so their own dependencies (e.g. libcurl ->
                # libz) still resolve from the system instead of our
                # bundled copies. Force $ORIGIN so cross-references between
                # these flat top-level libs stay self-contained. (Does NOT
                # touch the ROCm closure files bundled above -- those keep
                # their own carefully-relative vendor RPATHs, which is why
                # we preserved ROCm's subdirectory layout in the first
                # place.)
                if [[ -f "$m" && ! -L "$m" ]]; then
                    patchelf --set-rpath '$ORIGIN' "$LIB_DIR/$(basename "$m")" 2>/dev/null || true
                fi
            done <<< "$matches"
            return 0
        fi
    done
    fail "$pattern not found anywhere searched -- package would be non-self-contained"
}
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
Description: llama.cpp + TurboQuant ROCm/HIP backend (AMD Radeon 780M / AOCC Zen4)
 Self-contained ROCm 10.0 HIP-accelerated build of llama-cpp-turboquant for
 the AMD Radeon 780M iGPU (gfx1103, RDNA3), compiled with AMD AOCC 5.2.0 for
 Zen4 CPU code and ROCm's amdclang for HIP device kernels, with rocWMMA
 flash attention. Bundles the full ROCm runtime library closure (HIP, hipBLAS,
 rocBLAS, rocSOLVER, ROCr, rocroller/LLVM JIT backend, etc.) discovered via
 ldd, preserving ROCm's own relative library layout so vendor RPATHs resolve
 without patching.
 .
 Requirements (host system): the amdgpu kernel driver for gfx1103. No ROCm
 userspace packages required -- everything is bundled.
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
# that every non-glibc dependency resolves from OUR lib/ (subdirectories
# like lib/rocm_sysdeps/lib/ and lib/llvm/lib/ still count -- $LIB_DIR is
# their common prefix). No held-back host libs expected here: unlike CUDA's
# libcuda.so, ROCm's HSA runtime talks to the amdgpu kernel driver directly
# via ioctl, not through a separate userspace .so, so this package should be
# fully self-contained.
echo "[4/4] Verifying package..."
dpkg-deb -I "$DEB_FILE" > /dev/null && echo "  package structure OK"
unset LD_LIBRARY_PATH

LIB_DIR_REAL=$(realpath -m "$LIB_DIR")
ALLOWED_HOST_LIBS='^(linux-vdso\.so|ld-linux-x86-64\.so|libc\.so|libm\.so|libpthread\.so|libdl\.so|librt\.so)'
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
[[ $NOT_SELF_CONTAINED -eq 0 ]] && echo "  all libraries resolve from the bundled lib/ (or are standard glibc primitives)" || fail "package is not self-contained -- see above"

echo ""
echo "=== Build complete: $DEB_FILE ($(du -h "$DEB_FILE" | cut -f1)) ==="
echo "Install: sudo dpkg -i $DEB_FILE"
echo "Binary after install: $INSTALL_PREFIX/bin/llama-cli"
