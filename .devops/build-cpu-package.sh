#!/usr/bin/env bash
#
# Build the self-contained vai-llama.cpp-CPU .deb package.
#
# Reuses the already-verified x64-linux-cpu-aocc-release preset (AOCC 5.2.0,
# GGML_NATIVE=ON -- Zen4 AVX-512/VNNI tuned for this exact Ryzen 7 8845HS,
# no GPU backend) rather than a separate hand-rolled cmake flag set.
#
# Usage:
#   ./.devops/build-cpu-package.sh
#
# Must be run from the repository root.

set -euo pipefail

# A stale LD_LIBRARY_PATH inherited from the invoking shell (e.g. leftover
# ROCm dev-environment exports) takes priority over RUNPATH in glibc's
# search order and can silently corrupt library resolution / verification
# below. Build and verify against a clean environment.
unset LD_LIBRARY_PATH

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PRESET="x64-linux-cpu-aocc-release"
BUILD_DIR="build-${PRESET}"
STAGING_DIR="staging-cpu"
INSTALL_PREFIX="/opt/vai-llama/cpu"
PACKAGE_NAME="vai-llama.cpp-CPU"
VERSION="0.0.$(git rev-list --count HEAD)"
AOCC_PATH="${AOCC_PATH:-/opt/AMD/aocc-compiler-5.2.0}"

fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

[[ -d "$BUILD_DIR" ]] || fail "'$BUILD_DIR' does not exist — run ./va-make-CPU-aocc.sh make first"

echo "=== $PACKAGE_NAME package builder ==="
echo "Version: $VERSION"
echo "Install prefix: $INSTALL_PREFIX"
echo ""

# Step 1: install the already-built binaries into the staging tree
echo "[1/4] Installing $PRESET build into staging..."
rm -rf "$STAGING_DIR"
cmake --install "$BUILD_DIR" --prefix "$STAGING_DIR$INSTALL_PREFIX"

# Step 2: bundle system runtime libraries
echo "[2/4] Bundling system runtime libraries..."
LIB_DIR="$STAGING_DIR$INSTALL_PREFIX/lib"
mkdir -p "$LIB_DIR"

bundle_lib() {
    local pattern="$1"
    local found
    found=$(find "$LIB_DIR" -maxdepth 1 -name "$pattern" 2>/dev/null | head -1)
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
                # Raw system libraries (libcurl, libssl, ...) carry no
                # $ORIGIN-relative RPATH of their own, so their own
                # dependencies (e.g. libcurl -> libz) still resolve from the
                # system instead of our bundled copies. Force $ORIGIN so
                # cross-references between bundled libs stay self-contained.
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
Description: llama.cpp + TurboQuant CPU-only backend (Ryzen 7 8845HS / AOCC Zen4)
 Self-contained CPU-only build of llama-cpp-turboquant, compiled natively
 (-march=native) with AMD AOCC 5.2.0 for this exact Ryzen 7 8845HS (Zen4,
 AVX-512/VNNI). No GPU backend. Native-optimized for this specific CPU, not
 a portable multi-dispatch build -- will not run correctly on a different
 x86_64 microarchitecture.
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
# bundled copy (a real bug found while building this package). Check that
# every non-glibc dependency resolves from OUR lib/, not some host path.
echo "[4/4] Verifying package..."
dpkg-deb -I "$DEB_FILE" > /dev/null && echo "  package structure OK"

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
    # ldd prints unnormalized paths (e.g. bin/../lib/libfoo.so) -- resolve
    # both sides before comparing, or every bundled lib false-positives here.
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
