# vai-llama-vulkan Package — Self-Contained Vulkan Installation

**Updated:** 2026-06-08  
**Package:** `vai-llama-vulkan`  
**Install Location:** `/opt/vai-llama/vulkan`  
**Build Script:** `./devops/build-vulkan-package.sh`

---

## Overview

The `vai-llama-vulkan` package provides a **self-contained, zero-runtime-dependency** installation of llama-cpp-turboquant with Vulkan GPU acceleration. All required runtime libraries (Vulkan runtime, OpenGL ES, system libraries) are bundled within the package.

### Key Features

- **Zero Runtime Dependencies**: No need to install Vulkan, Mesa, or other GPU libraries on the host system
- **Self-Contained**: All dependencies bundled in the package
- **Multiple GPU Support**: Works with AMD (RADV), Intel (ANV), and other Vulkan-compatible GPUs
- **Complete Toolset**: Includes `llama.vulkan`, `llama-server.vulkan`, `llama-bench.vulkan`, and all utilities

---

## Installation

### Prerequisites

- Linux x86_64 system
- Build tools: `cmake`, `ninja`, `gcc`, `make`
- AOCC 5.14 toolchain (for building): `/opt/AMD/aocc-compiler-5.14.0`
- Vulkan runtime libraries (for bundling): `libvulkan1`, `libglapi0`, `libEGL1`, `libGLESv2`

### Build the Package

```bash
# Clone the repository
git clone https://github.com/your-org/llama-cpp-turboquant.git
cd llama-cpp-turboquant

# Build the Vulkan package
./devops/build-vulkan-package.sh

# The package will be created as:
# vai-llama-vulkan_<version>_amd64.deb
```

### Install the Package

```bash
# Copy the package to your system
sudo cp vai-llama-vulkan_*.deb /usr/local/

# Install the package
sudo dpkg -i /usr/local/vai-llama-vulkan_*.deb

# Fix any dependency issues (should be none)
sudo apt-get install -f
```

### Verify Installation

```bash
# Check installation
ls -la /opt/vai-llama/vulkan/

# Verify binaries
/opt/vai-llama/vulkan/bin/llama.vulkan --version
/opt/vai-llama/vulkan/bin/llama-server.vulkan --version

# Test with a model
/opt/vai-llama/vulkan/bin/llama.vulkan -m /path/to/model.gguf -p "Hello, World!"
```

---

## Package Contents

### Installation Structure

```
/opt/vai-llama/vulkan/
├── bin/
│   ├── llama.vulkan         # Command-line interface (Vulkan)
│   ├── llama-server.vulkan  # REST API server (Vulkan)
│   ├── llama-bench.vulkan   # Benchmark tool (Vulkan)
│   ├── llama-quantize.vulkan # Model quantization tool (Vulkan)
│   └── ...                  # Other utilities
├── lib/
│   ├── libllama.so.*        # Main library
│   ├── libggml-vulkan.so.*  # Vulkan backend
│   ├── libggml-cpu.so.*     # CPU backend
│   ├── libcurl.so.*         # HTTP client
│   ├── libssl.so.*          # SSL/TLS
│   ├── libvulkan.so.1       # Vulkan runtime (bundled)
│   ├── libglapi.so.*        # OpenGL API (bundled)
│   ├── libEGL.so.*          # EGL (bundled)
│   ├── libGLESv2.so.*       # OpenGL ES (bundled)
│   ├── libglvnd.so.*        # GLVND (bundled)
│   ├── libgomp.so.*         # OpenMP runtime (bundled)
│   ├── libstdc++.so.*       # C++ standard library (bundled)
│   └── libgcc_s.so.*        # GCC runtime (bundled)
└── share/
    └── llama/               # Documentation and man pages
```

### Bundled Libraries

#### Vulkan Runtime Libraries
- `libvulkan.so.1` - Vulkan ICD loader
- `libglapi.so.*` - OpenGL API implementation
- `libEGL.so.*` - EGL library
- `libGLESv2.so.*` - OpenGL ES 2.0
- `libglvnd.so.*` - GL Vendor Neutral Dispatch

#### System Libraries
- `libgomp.so.*` - GNU OpenMP runtime
- `libstdc++.so.*` - C++ standard library
- `libgcc_s.so.*` - GCC runtime
- `libssl.so.*` - OpenSSL
- `libcrypto.so.*` - OpenSSL crypto
- `libcurl.so.*` - libcurl

---

## Usage Examples

### Command-Line Interface

```bash
# Basic chat
/opt/vai-llama/vulkan/bin/llama.vulkan -m /path/to/model.gguf -c

# Server mode
/opt/vai-llama/vulkan/bin/llama-server.vulkan -m /path/to/model.gguf -c 2048

# Benchmark
/opt/vai-llama/vulkan/bin/llama-bench.vulkan -m /path/to/model.gguf -ngl 99
```

### REST API Server

```bash
# Start server
/opt/vai-llama/vulkan/bin/llama-server.vulkan -m /path/to/model.gguf -c 2048 -t 8

# Test with curl
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model.gguf",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Build Options

### CMake Configuration

The build script uses the following CMake options:

```bash
-DCMAKE_BUILD_TYPE=Release
-DCMAKE_INSTALL_PREFIX=/opt/vai-llama/vulkan
-DCMAKE_INSTALL_RPATH="\$ORIGIN/../lib;\$ORIGIN"
-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
-DGGML_NATIVE=OFF
-DGGML_VULKAN=ON
-DGGML_CUDA=OFF
-DGGML_HIP=OFF
-DGGML_METAL=OFF
-DGGML_BACKEND_DL=ON
-DGGML_CPU_ALL_VARIANTS=ON
-DLLAMA_BUILD_TESTS=OFF
-DLLAMA_BUILD_EXAMPLES=OFF
-DLLAMA_BUILD_TOOLS=ON
-DLLAMA_BUILD_SERVER=ON
-DLLAMA_OPENSSL=ON
-DLLAMA_BIN_NAME_SUFFIX=vulkan
```

### GPU Backend Selection

The package is built with **Vulkan only**:
- `GGML_VULKAN=ON` - Enable Vulkan backend
- `GGML_CUDA=OFF` - Disable CUDA
- `GGML_HIP=OFF` - Disable HIP/ROCm
- `GGML_METAL=OFF` - Disable Metal

---

## Troubleshooting

### Missing Libraries

If you encounter "library not found" errors:

```bash
# Check for missing libraries
ldd /opt/vai-llama/vulkan/bin/llama.vulkan

# Missing libraries should be bundled
# If still missing, check the package contents
dpkg -L vai-llama-vulkan | grep lib
```

### GPU Detection

```bash
# Check Vulkan devices
vulkaninfo --summary

# Check if RADV is available
ls /usr/share/vulkan/icd.d/radeon_icd.x86_64.json
```

### Performance Issues

```bash
# Check GPU support
vulkaninfo | grep -i "physical device"

# Check Vulkan layers
vulkaninfo --layers
```

---

## Package Size

Typical package sizes:

| Component | Size |
|-----------|------|
| Binaries | ~15 MB |
| Libraries | ~80 MB |
| **Total** | **~95 MB** |

---

## License

This package is built from the llama-cpp-turboquant project. See the project's LICENSE file for details.

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/your-org/llama-cpp-turboquant/issues
- Documentation: https://github.com/your-org/llama-cpp-turboquant/docs
