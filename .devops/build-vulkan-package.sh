#!/usr/bin/env bash
#
# Build the Vulkan-only self-contained .deb package for llama-cpp-turboquant
# 
# This script builds a complete, self-contained Debian package with all runtime
# dependencies bundled. The resulting package installs to /opt/vai-llama and
# requires no additional GPU library packages beyond the host's kernel-level
# Vulkan driver.
#
# Usage:
#   ./build-vulkan-package.sh [--version VERSION]
#

set -euo pipefail

# Default values
VERSION="${1:-}"
INSTALL_PREFIX="/opt/vai-llama/vulkan"
PACKAGE_NAME="vai-llama-vulkan"
BUILD_DIR="build-vulkan-package"
STAGING_DIR="staging-vulkan"

# Parse version argument
if [ "${VERSION:-}" = "--version" ]; then
  VERSION="0.0.$(git rev-list --count HEAD)"
elif [ -n "$VERSION" ]; then
  : # Use provided version
else
  VERSION="0.0.$(git rev-list --count HEAD)"
fi

echo "=== VAI-LLAMA Vulkan Package Builder ==="
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
  -DGGML_VULKAN=ON \
  -DGGML_CUDA=OFF \
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
  -DCMAKE_TOOLCHAIN_FILE="${sourceDir}/cmake/x64-linux-vulkan-aocc-54.cmake" \
  -DLLAMA_BIN_NAME_SUFFIX="vulkan"

# Step 2: Build the project
echo "[2/6] Building llama-cpp with Vulkan backend..."
cmake --build "$BUILD_DIR" --config Release -j$(nproc)

# Step 3: Install to staging area
echo "[3/6] Installing to staging directory..."
cmake --install "$BUILD_DIR" --config Release --prefix "$STAGING_DIR$INSTALL_PREFIX"

# Step 4: Bundle Vulkan runtime libraries
echo "[4/6] Bundling Vulkan runtime libraries..."

VULKAN_LIBS=(
  "libvulkan.so.1"
  "libglslang*.*"
  "libSPIRV*.*"
)

for lib in "${VULKAN_LIBS[@]}"; do
  # Search for the library in installed location first, then system paths
  found=""
  
  # Check if already installed (should be from cmake install)
  installed=$(find "$STAGING_DIR$INSTALL_PREFIX/lib" -name "$lib" -type f 2>/dev/null | head -1)
  if [ -n "$installed" ]; then
    echo "  Found $lib at: $installed"
    continue
  fi
  
  # Search system paths for Vulkan libraries
  for path in /usr/lib/x86_64-linux-gnu /usr/lib64; do
    libpath=$(find "$path" -name "${lib}" -type f 2>/dev/null | head -1)
    if [ -n "$libpath" ]; then
      echo "  Found $lib at: $libpath"
      cp -P "$libpath" "$STAGING_DIR$INSTALL_PREFIX/lib/"
      found=1
      break
    fi
  done
  
  # Fallback for glslang and SPIRV which may be in different locations
  if [ -z "$found" ] && [[ "$lib" == *"glslang"* ]]; then
    libpath=$(find /usr/lib/x86_64-linux-gnu -name "libGLSLANG*.*" -type f 2>/dev/null | head -1)
    if [ -n "$libpath" ]; then
      echo "  Found glslang at: $libpath"
      cp -P "$libpath" "$STAGING_DIR$INSTALL_PREFIX/lib/"
      found=1
    fi
  fi
  
  # Fallback for SPIRV
  if [ -z "$found" ] && [[ "$lib" == *"SPIRV"* ]]; then
    libpath=$(find /usr/lib/x86_64-linux-gnu -name "libSPIR-V*.*" -type f 2>/dev/null | head -1)
    if [ -n "$libpath" ]; then
      echo "  Found SPIRV at: $libpath"
      cp -P "$libpath" "$STAGING_DIR$INSTALL_PREFIX/lib/"
      found=1
    fi
  fi
  
  # Check for OpenCL (Vulkan dependency)
  if [ -z "$found" ] && [[ "$lib" == *"OpenCL"* ]]; then
    libpath=$(find /usr/lib/x86_64-linux-gnu -name "libOpenCL*.*" -type f 2>/dev/null | head -1)
    if [ -n "$libpath" ]; then
      echo "  Found OpenCL at: $libpath"
      cp -P "$libpath" "$STAGING_DIR$INSTALL_PREFIX/lib/"
      found=1
    fi
  fi
  
  # Check for GLES (Vulkan dependency)
  if [ -z "$found" ] && [[ "$lib" == *"GLES"* ]]; then
    libpath=$(find /usr/lib/x86_64-linux-gnu -name "libEGL*.*" -type f 2>/dev/null | head -1)
    if [ -n "$libpath" ]; then
      echo "  Found EGL at: $libpath"
      cp -P "$libpath" "$STAGING_DIR$INSTALL_PREFIX/lib/"
      found=1
      
      libpath=$(find /usr/lib/x86_64-linux-gnu -name "libGLESv2*.*" -type f 2>/dev/null | head -1)
      if [ -n "$libpath" ]; then
        echo "  Found GLESv2 at: $libpath"
        cp -P "$libpath" "$STAGING_DIR$INSTALL_PREFIX/lib/"
        found=1
      fi
    fi
  fi
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
vai-llama-vulkan ($VERSION) unstable; urgency=medium

  * Initial self-contained Vulkan package release.
  * Bundles all GPU runtime libraries for zero-runtime-dependency installation.
  * Includes: llama-cli, llama-server, llama-bench, and all tools.

 -- VAI Team <team@vai.example.com>  $(date +%Y-%m-%d)

EOF

# Create the package
dpkg-deb \
  --root-owner-group \
  --set-sequence 100 \
  -b "vai-llama-vulkan_${VERSION}_amd64" \
  "vai-llama-vulkan_${VERSION}_amd64.deb"

cd ..

# Verify the package
echo ""
echo "=== Package Verification ==="
dpkg-deb -I vai-llama-vulkan_${VERSION}_amd64.deb > /dev/null && echo "✓ Package structure is valid" || echo "✗ Package verification failed"

ldd "$STAGING_DIR$INSTALL_PREFIX/bin/llama.vulkan" 2>/dev/null | grep -E "(not found|=>)" || echo "✓ No missing shared libraries detected (except expected host libs)"

echo ""
echo "=== Build Complete ==="
echo "Package: vai-llama-vulkan_${VERSION}_amd64.deb"
echo "Location: $(pwd)/vai-llama-vulkan_${VERSION}_amd64.deb"
echo "Size: $(du -h vai-llama-vulkan_${VERSION}_amd64.deb | cut -f1)"
echo ""
echo "To install:"
echo "  sudo dpkg -i vai-llama-vulkan_${VERSION}_amd64.deb"
echo "  sudo apt-get install -f  # Fix any dependency issues"
echo ""
echo "Installation path: $INSTALL_PREFIX"
echo "Binary: $INSTALL_PREFIX/bin/llama.vulkan"
