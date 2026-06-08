# VAI-LLAMA Packaging Plan

**Created:** 2026-06-08  
**Project:** llama-cpp-turboquant  
**Target Installation:** `/opt/vai-llama`

---

## Executive Summary

This document outlines the plan to create two self-contained Debian packages for llama-cpp-turboquant:

1. **vai-llama-vulkan** - Vulkan-only GPU acceleration package
2. **vai-llama-cuda** - NVIDIA CUDA-only GPU acceleration package

Both packages will be **zero-runtime-dependency**, meaning all required libraries are bundled within the package, allowing installation on any Linux x86_64 system without requiring pre-installed GPU libraries.

---

## Package Specifications

### Package 1: vai-llama-vulkan

| Attribute | Value |
|-----------|-------|
| **Package Name** | `vai-llama-vulkan` |
| **Install Location** | `/opt/vai-llama/vulkan` |
| **GPU Backend** | Vulkan (RADV, ANV, etc.) |
| **Target GPUs** | AMD (RADV), Intel (ANV), other Vulkan-compatible GPUs |
| **Estimated Size** | ~95 MB |
| **Build Script** | `.devops/build-vulkan-package.sh` |
| **CMake Preset** | `x64-linux-vulkan-aocc.cmake` |
| **Binary Name** | `llama.vulkan` |

### Package 2: vai-llama-cuda

| Attribute | Value |
|-----------|-------|
| **Package Name** | `vai-llama-cuda` |
| **Install Location** | `/opt/vai-llama/cuda` |
| **GPU Backend** | CUDA (NVIDIA) |
| **Target GPUs** | NVIDIA (RTX 4060, A100, H100, etc.) |
| **Estimated Size** | ~215 MB |
| **Build Script** | `.devops/build-cuda-package.sh` |
| **CMake Preset** | `x64-linux-cuda-nvidia-rtx4060.cmake` |
| **Binary Name** | `llama.cuda` |

---

## Installation Structure

### Vulkan Package: `/opt/vai-llama/vulkan/`

```
/opt/vai-llama/vulkan/
├── bin/
│   ├── llama.vulkan       # Command-line interface (Vulkan)
│   ├── llama-server.vulkan # REST API server (Vulkan)
│   ├── llama-bench.vulkan # Benchmark tool (Vulkan)
│   ├── llama-quantize.vulkan # Model quantization tool (Vulkan)
│   └── ...                # Other utilities
├── lib/
│   ├── libllama.so.*      # Main library
│   ├── libggml-vulkan.so.* # Vulkan backend
│   ├── libggml-cpu.so.*   # CPU backend
│   ├── libcurl.so.*       # HTTP client
│   ├── libssl.so.*        # SSL/TLS
│   ├── libvulkan.so.1     # Vulkan runtime (bundled)
│   ├── libglapi.so.*      # OpenGL API (bundled)
│   ├── libEGL.so.*        # EGL (bundled)
│   ├── libGLESv2.so.*     # OpenGL ES (bundled)
│   ├── libglvnd.so.*      # GLVND (bundled)
│   ├── libgomp.so.*       # OpenMP runtime (bundled)
│   ├── libstdc++.so.*     # C++ standard library (bundled)
│   └── libgcc_s.so.*      # GCC runtime (bundled)
└── share/
    └── llama/             # Documentation and man pages
```

### CUDA Package: `/opt/vai-llama/cuda/`

```
/opt/vai-llama/cuda/
├── bin/
│   ├── llama.cuda         # Command-line interface (CUDA)
│   ├── llama-server.cuda  # REST API server (CUDA)
│   ├── llama-bench.cuda   # Benchmark tool (CUDA)
│   ├── llama-quantize.cuda # Model quantization tool (CUDA)
│   └── ...                # Other utilities
├── lib/
│   ├── libllama.so.*      # Main library
│   ├── libggml-cuda.so.*  # CUDA backend
│   ├── libggml-cpu.so.*   # CPU backend
│   ├── libcurl.so.*       # HTTP client
│   ├── libssl.so.*        # SSL/TLS
│   ├── libcudart.so.*     # CUDA runtime (bundled)
│   ├── libnvrtc.so.*      # CUDA NVRTC (bundled)
│   ├── libcublas.so.*     # cuBLAS (bundled)
│   ├── libcublasLt.so.*   # cuBLASLt (bundled)
│   ├── libcusparse.so.*   # cuSPARSE (bundled)
│   ├── libnvjpeg.so.*     # NVIDIA JPEG (bundled)
│   ├── libgomp.so.*       # OpenMP runtime (bundled)
│   ├── libstdc++.so.*     # C++ standard library (bundled)
│   └── libgcc_s.so.*      # GCC runtime (bundled)
└── share/
    └── llama/             # Documentation and man pages
```

---

## Build Process

### Step 1: Configure Build

**Vulkan Package:**
```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/vai-llama/vulkan \
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
  -DCMAKE_TOOLCHAIN_FILE=cmake/x64-linux-vulkan-aocc.cmake \
  -DLLAMA_BIN_NAME_SUFFIX=vulkan
```

**CUDA Package:**
```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/vai-llama/cuda \
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
  -DCMAKE_TOOLCHAIN_FILE=cmake/x64-linux-cuda-nvidia-rtx4060.cmake \
  -DLLAMA_BIN_NAME_SUFFIX=cuda
```

### Step 2: Build

```bash
cmake --build build --config Release -j$(nproc)
```

### Step 3: Install to Staging

```bash
cmake --install build --config Release --prefix=/opt/vai-llama/vulkan  # Vulkan
cmake --install build --config Release --prefix=/opt/vai-llama/cuda    # CUDA
```

### Step 4: Bundle Dependencies

Copy required runtime libraries from system paths:

**Vulkan Package:**
- `libvulkan.so.1`
- `libglapi.so.*`
- `libEGL.so.*`
- `libGLESv2.so.*`
- `libglvnd.so.*`
- `libgomp.so.*`
- `libstdc++.so.*`
- `libgcc_s.so.*`
- `libssl.so.*`
- `libcrypto.so.*`
- `libcurl.so.*`

**CUDA Package:**
- `libcudart.so.*`
- `libnvrtc.so.*`
- `libcurand.so.*`
- `libcublas.so.*`
- `libcublasLt.so.*`
- `libcusparse.so.*`
- `libnvjpeg.so.*`
- `libgomp.so.*`
- `libstdc++.so.*`
- `libgcc_s.so.*`
- `libssl.so.*`
- `libcrypto.so.*`
- `libcurl.so.*`

### Step 5: Create Debian Package

```bash
cd staging
dpkg-deb --root-owner-group --set-sequence 100 \
  -b "vai-llama-<gpu>_<version>_amd64" \
  "vai-llama-<gpu>_<version>_amd64.deb"
```

---

## CMake Configuration Details

### Vulkan Package CMake Options

```cmake
set(GGML_VULKAN ON)
set(GGML_CUDA OFF)
set(GGML_HIP OFF)
set(GGML_METAL OFF)
set(GGML_BACKEND_DL ON)
set(GGML_CPU_ALL_VARIANTS ON)
set(LLAMA_BIN_NAME_SUFFIX "vulkan")
```

### CUDA Package CMake Options

```cmake
set(GGML_CUDA ON)
set(GGML_VULKAN OFF)
set(GGML_HIP OFF)
set(GGML_METAL OFF)
set(GGML_BACKEND_DL ON)
set(GGML_CPU_ALL_VARIANTS ON)
set(LLAMA_BIN_NAME_SUFFIX "cuda")
```

---

## Dependency Analysis

### Vulkan Package Dependencies

| Library | Purpose | Bundled |
|---------|---------|---------|
| libvulkan.so.1 | Vulkan ICD loader | ✓ |
| libglapi.so.* | OpenGL API | ✓ |
| libEGL.so.* | EGL library | ✓ |
| libGLESv2.so.* | OpenGL ES 2.0 | ✓ |
| libglvnd.so.* | GLVND | ✓ |
| libgomp.so.* | OpenMP runtime | ✓ |
| libstdc++.so.* | C++ standard library | ✓ |
| libgcc_s.so.* | GCC runtime | ✓ |
| libssl.so.* | OpenSSL | ✓ |
| libcrypto.so.* | OpenSSL crypto | ✓ |
| libcurl.so.* | libcurl | ✓ |

### CUDA Package Dependencies

| Library | Purpose | Bundled |
|---------|---------|---------|
| libcudart.so.* | CUDA Runtime | ✓ |
| libnvrtc.so.* | NVRTC | ✓ |
| libcurand.so.* | Random numbers | ✓ |
| libcublas.so.* | Linear algebra | ✓ |
| libcublasLt.so.* | cuBLASLt | ✓ |
| libcusparse.so.* | Sparse linear algebra | ✓ |
| libnvjpeg.so.* | JPEG library | ✓ |
| libgomp.so.* | OpenMP runtime | ✓ |
| libstdc++.so.* | C++ standard library | ✓ |
| libgcc_s.so.* | GCC runtime | ✓ |
| libssl.so.* | OpenSSL | ✓ |
| libcrypto.so.* | OpenSSL crypto | ✓ |
| libcurl.so.* | libcurl | ✓ |

---

## Build Scripts

### Vulkan Package Build Script

**Location:** `.devops/build-vulkan-package.sh`

**Features:**
- Configures CMake with Vulkan backend
- Builds llama-cpp-turboquant with Vulkan support
- Installs to `/opt/vai-llama/vulkan`
- Bundles Vulkan runtime libraries
- Bundles system libraries
- Creates Debian package
- Verifies package structure

### CUDA Package Build Script

**Location:** `.devops/build-cuda-package.sh`

**Features:**
- Configures CMake with CUDA backend
- Builds llama-cpp-turboquant with CUDA support
- Installs to `/opt/vai-llama/cuda`
- Bundles CUDA runtime libraries
- Bundles system libraries
- Creates Debian package
- Verifies package structure

---

## Testing Strategy

### Package Verification

```bash
# Verify package structure
dpkg-deb -I vai-llama-<gpu>_<version>_amd64.deb

# Check for missing libraries
ldd /opt/vai-llama/<gpu>/bin/llama.<gpu>

# Verify binaries
/opt/vai-llama/<gpu>/bin/llama.<gpu> --version
/opt/vai-llama/<gpu>/bin/llama-server.<gpu> --version
```

### Runtime Testing

```bash
# Test CLI
/opt/vai-llama/<gpu>/bin/llama.<gpu> -m /path/to/model.gguf -p "Hello, World!"

# Test server
/opt/vai-llama/<gpu>/bin/llama-server.<gpu> -m /path/to/model.gguf -c 2048 &
curl http://localhost:8080/v1/chat/completions
```

---

## Installation Verification

### Verify Installation

```bash
# Check installation directory
ls -la /opt/vai-llama/vulkan/
ls -la /opt/vai-llama/cuda/

# Verify binaries
file /opt/vai-llama/vulkan/bin/llama.vulkan
file /opt/vai-llama/cuda/bin/llama.cuda

# Check library dependencies
ldd /opt/vai-llama/vulkan/bin/llama.vulkan
ldd /opt/vai-llama/cuda/bin/llama.cuda

# Verify GPU backend
/opt/vai-llama/vulkan/bin/llama.vulkan -m /path/to/model.gguf --info
/opt/vai-llama/cuda/bin/llama.cuda -m /path/to/model.gguf --info
```

### Expected Output

```
/opt/vai-llama/vulkan/bin/llama.vulkan: ELF 64-bit LSB executable, x86-64, ...
/opt/vai-llama/cuda/bin/llama.cuda: ELF 64-bit LSB executable, x86-64, ...
```

---

## Rollback Plan

### Remove Package

```bash
# Remove package
sudo dpkg -r vai-llama-vulkan
sudo dpkg -r vai-llama-cuda

# Remove installation directory
sudo rm -rf /opt/vai-llama/vulkan
sudo rm -rf /opt/vai-llama/cuda
```

### Restore from Backup

```bash
# Restore from backup
sudo dpkg -i /backup/vai-llama-<gpu>_<version>_amd64.deb
sudo apt-get install -f
```

---

## Maintenance

### Update Package

```bash
# Pull latest changes
git pull origin main

# Rebuild package
./devops/build-vulkan-package.sh --version 1.0.1
./devops/build-cuda-package.sh --version 1.0.1
```

### Monitor Dependencies

```bash
# Check for library updates
ldd /opt/vai-llama/vulkan/bin/llama.vulkan | grep -E "(not found|=>)"
ldd /opt/vai-llama/cuda/bin/llama.cuda | grep -E "(not found|=>)"

# Monitor package size
du -sh /opt/vai-llama/vulkan
du -sh /opt/vai-llama/cuda
```

---

## Known Limitations

### Vulkan Package

1. Requires Vulkan-compatible GPU driver
2. Does not support NVIDIA GPUs (no Vulkan driver)
3. May have performance variations across GPU vendors

### CUDA Package

1. Requires NVIDIA GPU
2. Does not support AMD or Intel GPUs
3. CUDA runtime libraries are large (~150 MB)

---

## Future Enhancements

1. **Multi-GPU Support**: Create packages that support both Vulkan and CUDA
2. **ARM Support**: Create packages for ARM64 architectures
3. **Container Support**: Create Docker images with pre-bundled dependencies
4. **Snap Package**: Create Snap packages for easier distribution
5. **Flatpak Support**: Create Flatpak packages for sandboxed installation

---

## References

- [llama-cpp-turboquant Repository](https://github.com/your-org/llama-cpp-turboquant)
- [Vulkan Documentation](https://www.khronos.org/vulkan/)
- [CUDA Documentation](https://docs.nvidia.com/cuda/)
- [Debian Package Format](https://www.debian.org/doc/debian-policy/)

---

## Appendix

### Build Command Summary

**Vulkan Package:**
```bash
./devops/build-vulkan-package.sh
```

**CUDA Package:**
```bash
./devops/build-cuda-package.sh
```

### Package Naming Convention

```
vai-llama-<gpu>_<version>_amd64.deb
```

Where:
- `<gpu>`: `vulkan` or `cuda`
- `<version>`: Version number (e.g., `0.0.123`)
- `amd64`: Architecture (x86_64)

### Installation Command

```bash
sudo dpkg -i vai-llama-<gpu>_<version>_amd64.deb
sudo apt-get install -f
```

### Binary Names

| Package | Binary | Path |
|---------|--------|------|
| vai-llama-vulkan | `llama.vulkan` | `/opt/vai-llama/vulkan/bin/llama.vulkan` |
| vai-llama-vulkan | `llama-server.vulkan` | `/opt/vai-llama/vulkan/bin/llama-server.vulkan` |
| vai-llama-vulkan | `llama-bench.vulkan` | `/opt/vai-llama/vulkan/bin/llama-bench.vulkan` |
| vai-llama-cuda | `llama.cuda` | `/opt/vai-llama/cuda/bin/llama.cuda` |
| vai-llama-cuda | `llama-server.cuda` | `/opt/vai-llama/cuda/bin/llama-server.cuda` |
| vai-llama-cuda | `llama-bench.cuda` | `/opt/vai-llama/cuda/bin/llama-bench.cuda` |
