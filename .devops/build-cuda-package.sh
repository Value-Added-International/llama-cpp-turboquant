#!/usr/bin/env bash
#
# Build the CUDA-only self-contained .deb package for llama-cpp-turboquant
# 
# This script builds a complete, self-contained Debian package with all runtime
# dependencies bundled. The resulting package installs to /opt/vai-llama and
# requires no additional GPU library packages beyond the host's CUDA driver.
#
# Usage:
#   ./build-cuda-package.sh [--version VERSION]
#

set -euo pipefail

# Default values
VERSION="${1:-}"
INSTALL_PREFIX="/opt/vai-llama/cuda"
PACKAGE_NAME="vai-llama-cuda"
BUILD_DIR="build-cuda-package"
STAGING_DIR="staging-cuda"

# Parse version argument
if [ "${VERSION:-}" = "--version" ]; then
  VERSION="0.0.$(git rev-list --count HEAD)"
elif [ -n "$VERSION" ]; then
  : # Use provided version
else
  VERSION="0.0.$(git rev-list --count HEAD)"
fi

echo "=== VAI-LLAMA CUDA Package Builder ==="
echo "Version: $VERSION"
echo "Install prefix: $INSTALL_PREFIX"
echo ""

# Create build directory
rm -rf "$BUILD_DIR" "$STAGING_DIR"
mkdir -p "$BUILD_DIR" "$STAGING_DIR$INSTALL_PREFIX"

# Step 1: Configure the build
echo "[1/6] Configuring CMake build..."
cmake -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  -DCMAKE_INSTALL_RPATH="\$ORIGIN/../lib;\$ORIGIN" \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_VULKAN=OFF \
  -DGGML_CUDA=ON \
  -DGGML_HIP=OFF \
  -DGGML_METAL=OFF \
  -DGGML_BACKEND_DL=ON \
  -DGGML_CPU_ALL_VARIANTS=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_OPENSSL=ON \
  -DCMAKE_EXE_LINKER_FLAGS='-Wl,--allow-shlib-undefined' \
  -DCMAKE_TOOLCHAIN_FILE="${sourceDir}/cmake/x64-linux-cuda-nvidia-rtx4060.cmake" \
  -DLLAMA_BIN_NAME_SUFFIX="cuda"

# Step 2: Build the project
echo "[2/6] Building llama-cpp with CUDA backend..."
cmake --build "$BUILD_DIR" --config Release -j$(nproc)

# Step 3: Install to staging area
echo "[3/6] Installing to staging directory..."
cmake --install "$BUILD_DIR" --config Release --prefix "$STAGING_DIR$INSTALL_PREFIX"

# Step 4: Bundle CUDA runtime libraries
echo "[4/6] Bundling CUDA runtime libraries..."

CUDA_LIBS=(
  "libcudart.so*"
  "libnvrtc.so*"
  "libcurand.so*"
  "libcublas.so*"
  "libcublasLt.so*"
  "libcusparse.so*"
  "libcusolver.so*"
  "libnvjpeg.so*"
  "libnvinfer.so*"
  "libnvparsers.so*"
  "libnvonnxruntime.so*"
)

for lib in "${CUDA_LIBS[@]}"; do
  # Search for the library in installed location first, then system paths
  found=""
  
  # Check if already installed (should be from cmake install)
  installed=$(find "$STAGING_DIR$INSTALL_PREFIX/lib" -name "$lib" -type f 2>/dev/null | head -1)
  if [ -n "$installed" ]; then
    echo "  Found $lib at: $installed"
    continue
  fi
  
  # Search system paths for CUDA libraries
  for path in /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu /usr/lib64; do
    if [ -d "$path" ]; then
      libpath=$(find "$path" -name "${lib}" -type f 2>/dev/null | head -1)
      if [ -n "$libpath" ]; then
        echo "  Found $lib at: $libpath"
        cp -P "$libpath" "$STAGING_DIR$INSTALL_PREFIX/lib/"
        found=1
        break
      fi
    fi
  done
done

# Step 5: Bundle required system libraries
echo "[5/6] Bundling required system libraries..."

SYSTEM_LIBS=(
  "libgomp.so.1"
  "libstdc++.so.*"
  "libgcc_s.so.*"
  "libssl.so*"
  "libcrypto.so*"
  "libcurl.so*"
)

for lib in "${SYSTEM_LIBS[@]}"; do
  found=$(find "$STAGING_DIR$INSTALL_PREFIX/lib" -name "${lib}" -type f 2>/dev/null | head -1)
  
  if [ -z "$found" ]; then
    # Search system paths
    for path in /usr/lib/x86_64-linux-gnu /usr/lib64; do
      libpath=$(find "$path" -name "${lib}" -type f 2>/dev/null | head -1)
      if [ -n "$libpath" ]; then
        echo "  Bundling $lib from: $libpath"
        cp -P "$libpath" "$STAGING_DIR$INSTALL_PREFIX/lib/"
        break
      fi
    done
  else
    echo "  Library already present: $found"
  fi
done

# Step 6: Create the .deb package
echo "[6/6] Creating Debian package..."

cd "$STAGING_DIR"

# Generate changelog entry
cat > "changelog" << EOF
vai-llama-cuda ($VERSION) unstable; urgency=medium

  * Initial self-contained CUDA package release.
  * Bundles all CUDA runtime libraries for zero-runtime-dependency installation.
  * Includes: llama-cli, llama-server, llama-bench, and all tools.

 -- VAI Team <team@vai.example.com>  $(date +%Y-%m-%d)

EOF

# Create the package
dpkg-deb \
  --root-owner-group \
  --set-sequence 100 \
  -b "vai-llama-cuda_${VERSION}_amd64" \
  "vai-llama-cuda_${VERSION}_amd64.deb"

cd ..

# Verify the package
echo ""
echo "=== Package Verification ==="
dpkg-deb -I vai-llama-cuda_${VERSION}_amd64.deb > /dev/null && echo "✓ Package structure is valid" || echo "✗ Package verification failed"

ldd "$STAGING_DIR$INSTALL_PREFIX/bin/llama.cuda" 2>/dev/null | grep -E "(not found|=>)" || echo "✓ No missing shared libraries detected (except expected host libs)"

echo ""
echo "=== Build Complete ==="
echo "Package: vai-llama-cuda_${VERSION}_amd64.deb"
echo "Location: $(pwd)/vai-llama-cuda_${VERSION}_amd64.deb"
echo "Size: $(du -h vai-llama-cuda_${VERSION}_amd64.deb | cut -f1)"
echo ""
echo "To install:"
echo "  sudo dpkg -i vai-llama-cuda_${VERSION}_amd64.deb"
echo "  sudo apt-get install -f  # Fix any dependency issues"
echo ""
echo "Installation path: $INSTALL_PREFIX"
echo "Binary: $INSTALL_PREFIX/bin/llama.cuda"
