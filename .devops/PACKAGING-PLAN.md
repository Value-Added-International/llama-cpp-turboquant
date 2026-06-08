# VAI-LLAMA Packaging Plan

## Overview

This document describes the plan to create two self-contained Debian packages for llama-cpp-turboquant that install to `/opt/vai-llama`. Each package contains all runtime dependencies, requiring only a compatible GPU driver on the target system.

---

## Package 1: Vulkan-only (`vai-llama-vulkan`)

### Target Systems
- AMD GPUs with Vulkan support (Radeon RX 7800 XT, etc.)
- Any Linux distribution with Vulkan drivers installed

### Build Configuration
```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DGGML_CUDA=OFF \
  -DGGML_HIP=OFF \
  -DGGML_METAL=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_OPENSSL=ON
```

### Toolchain
- **CPU Compiler**: AMD AOCC 5.4 (`/opt/AMD/aocc-compiler-5.4/bin`)
- **GPU Backend**: Vulkan (glslc for shader compilation)
- **CMake Preset**: `vai-llama-vulkan` (`.devops/cmake-preset-vulkan.json`)

### Bundled Dependencies
The package bundles:
1. Application binaries (`server`, `llama-bench`, etc.)
2. CPU libraries with AOCC optimizations
3. Vulkan runtime libraries from system paths:
   - `libvulkan.so.1`
   - `libglslang*.*` (shader compiler/runtime)
   - `libSPIRV*.*` (SPIR-V loader)

### Installation Structure
```
/opt/vai-llama/
├── bin/
│   ├── server           # Main inference server
│   ├── llama-bench      # Benchmark tool
│   └── ...              # Other tools
├── lib/
│   ├── libggml-vulkan*.*     # Vulkan GPU kernels
│   ├── libggml-cpu*.*        # CPU fallback
│   ├── libvulkan.so.1        # Bundled Vulkan runtime
│   ├── libglslang*.*         # Bundled shader compiler
│   └── libSPIRV*.*           # Bundled SPIR-V loader
├── share/vai-llama/        # Man pages, READMEs
└── libexec/vai-llama/      # Shared libraries
```

---

## Package 2: NVIDIA CUDA (`vai-llama-cuda`)

### Target Systems
- NVIDIA GPUs (any architecture from Pascal onwards)
- Any Linux distribution with NVIDIA driver installed

### Build Configuration
```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_VULKAN=OFF \
  -DGGML_HIP=OFF \
  -DGGML_METAL=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_OPENSSL=ON
```

### Toolchain
- **GPU Compiler**: `nvcc` (CUDA toolkit)
- **CPU Compiler**: System GCC/Clang
- **CMake Preset**: `vai-llama-cuda` (`.devops/cmake-preset-cuda.json`)

### Bundled Dependencies
The package bundles:
1. Application binaries (`server`, `llama-bench`, etc.)
2. GPU libraries with CUDA kernels
3. CUDA runtime libraries from system paths:
   - `libcudart.so.*` (CUDA runtime)
   - `libcufile.so.*` (NVMe file I/O, optional)

### Installation Structure
```
/opt/vai-llama/
├── bin/
│   ├── server           # Main inference server
│   ├── llama-bench      # Benchmark tool
│   └── ...              # Other tools
├── lib/
│   ├── libggml-cuda*.*    # CUDA GPU kernels
│   ├── libggml-cpu*.*     # CPU fallback
│   ├── libcudart.so.*     # Bundled CUDA runtime
│   └── libcufile.so.*     # Bundled CUDA file I/O (optional)
├── share/vai-llama/      # Man pages, READMEs
└── libexec/vai-llama/    # Shared libraries
```

---

## Build Scripts

### Vulkan Package Builder
**File**: `.devops/build-vulkan-package.sh`

Usage:
```bash
./build-vulkan-package.sh [--version VERSION]
```

The script performs:
1. CMake configuration with AOCC toolchain preset
2. Build with Ninja/Make
3. Install to staging directory
4. Bundle Vulkan runtime libraries from system paths
5. Create `.deb` package with proper dependencies

### CUDA Package Builder
**File**: `.devops/build-cuda-package.sh`

Usage:
```bash
./build-cuda-package.sh [--version VERSION]
```

The script performs:
1. CMake configuration with CUDA toolchain preset
2. Build with Ninja/Make
3. Install to staging directory
4. Bundle CUDA runtime libraries from system paths
5. Create `.deb` package with proper dependencies

---

## Package Control Files

### Vulkan Package Control (`.devops/debian/vulkan/control`)
```
Package: vai-llama-vulkan
Version: 0.0.1
Architecture: amd64
Maintainer: VAI Team <team@vai.dev>
Description: Self-contained llama-cpp-turboquant for AMD Vulkan GPUs
```

### CUDA Package Control (`.devops/debian/cuda/control`)
```
Package: vai-llama-cuda
Version: 0.0.1
Architecture: amd64
Maintainer: VAI Team <team@vai.dev>
Description: Self-contained llama-cpp-turboquant for NVIDIA CUDA GPUs
```

---

## Build Process Summary

### For Vulkan Package:
```bash
# Ensure AOCC is installed at /opt/AMD/aocc-compiler-5.4
export PATH=/opt/AMD/aocc-compiler-5.4/bin:$PATH

./build-vulkan-package.sh
# Result: vai-llama-vulkan_0.0.1_amd64.deb in .devops/debian/vulkan/
```

### For CUDA Package:
```bash
# Ensure CUDA toolkit is installed at /usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH

./build-cuda-package.sh
# Result: vai-llama-cuda_0.0.1_amd64.deb in .devops/debian/cuda/
```

### Installation on Target System:
```bash
# Install package
dpkg -i vai-llama-vulkan_*.deb
# or
dpkg -i vai-llama-cuda_*.deb

# Verify installation
ls /opt/vai-llama/bin/server
/opt/vai-llama/bin/server --help
```

---

## Files Created/Modified

| File | Purpose |
|------|---------|
| `cmake/x64-linux-vulkan-aocc-54.cmake` | AOCC 5.4 toolchain for Vulkan builds |
| `cmake/x64-linux-cuda-nvidia.cmake` | NVIDIA CUDA toolchain configuration |
| `.devops/build-vulkan-package.sh` | Vulkan package build script (updated) |
| `.devops/build-cuda-package.sh` | CUDA package build script (updated) |
| `.devops/cmake-preset-cuda.json` | CMake preset for CUDA builds |
| `.devops/cmake-preset-vulkan.json` | CMake preset for Vulkan builds |

---

## Notes

1. **Vulkan Package**: Requires AMD AOCC 5.4 toolchain to be installed on the build machine at `/opt/AMD/aocc-compiler-5.4`. The package itself does NOT include compiler toolchains, only runtime libraries.

2. **CUDA Package**: Requires NVIDIA CUDA toolkit to be installed on the build machine. The package bundles `libcudart.so` but relies on system drivers for GPU execution.

3. **Self-Contained Guarantee**: Both packages bundle all necessary runtime libraries, so they can be deployed to any compatible Linux distribution without requiring pre-installed toolchains or additional library packages beyond the host's GPU driver.

4. **Installation Path**: All packages install to `/opt/vai-llama` as specified in the user requirements.

---

## Testing Checklist

Before finalizing, verify:
- [ ] Vulkan package builds successfully with AOCC 5.4
- [ ] CUDA package builds successfully with system CUDA toolkit
- [ ] Bundled libraries have correct RPATH settings
- [ ] Package installs cleanly to `/opt/vai-llama`
- [ ] `ldd` shows no missing dependencies after installation
