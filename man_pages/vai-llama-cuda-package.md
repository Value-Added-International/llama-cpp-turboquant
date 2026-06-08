# vai-llama-cuda Package — Self-Contained CUDA Installation

**Updated:** 2026-06-08  
**Package:** `vai-llama-cuda`  
**Install Location:** `/opt/vai-llama/cuda`  
**Build Script:** `./devops/build-cuda-package.sh`

---

## Overview

The `vai-llama-cuda` package provides a **self-contained, zero-runtime-dependency** installation of llama-cpp-turboquant with CUDA GPU acceleration. All required runtime libraries (CUDA runtime, cuBLAS, cuDNN, etc.) are bundled within the package.

### Key Features

- **Zero Runtime Dependencies**: No need to install CUDA toolkit or additional GPU libraries on the host system
- **Self-Contained**: All dependencies bundled in the package
- **NVIDIA GPU Support**: Works with NVIDIA GPUs (RTX 4060, A100, H100, etc.)
- **Complete Toolset**: Includes `llama.cuda`, `llama-server.cuda`, `llama-bench.cuda`, and all utilities

---

## Installation

### Prerequisites

- Linux x86_64 system
- Build tools: `cmake`, `ninja`, `gcc`, `make`
- NVIDIA CUDA toolkit (for building): `/usr/local/cuda-12.x`
- CUDA runtime libraries (for bundling): `libcudart`, `libnvrtc`, `libcublas`, etc.

### Build the Package

```bash
# Clone the repository
git clone https://github.com/your-org/llama-cpp-turboquant.git
cd llama-cpp-turboquant

# Build the CUDA package
./devops/build-cuda-package.sh

# The package will be created as:
# vai-llama-cuda_<version>_amd64.deb
```

### Install the Package

```bash
# Copy the package to your system
sudo cp vai-llama-cuda_*.deb /usr/local/

# Install the package
sudo dpkg -i /usr/local/vai-llama-cuda_*.deb

# Fix any dependency issues (should be none)
sudo apt-get install -f
```

### Verify Installation

```bash
# Check installation
ls -la /opt/vai-llama/cuda/

# Verify binaries
/opt/vai-llama/cuda/bin/llama.cuda --version
/opt/vai-llama/cuda/bin/llama-server.cuda --version

# Test with a model
/opt/vai-llama/cuda/bin/llama.cuda -m /path/to/model.gguf -p "Hello, World!"
```

---

## Package Contents

### Installation Structure

```
/opt/vai-llama/cuda/
├── bin/
│   ├── llama.cuda           # Command-line interface (CUDA)
│   ├── llama-server.cuda    # REST API server (CUDA)
│   ├── llama-bench.cuda     # Benchmark tool (CUDA)
│   ├── llama-quantize.cuda  # Model quantization tool (CUDA)
│   └── ...                  # Other utilities
├── lib/
│   ├── libllama.so.*        # Main library
│   ├── libggml-cuda.so.*    # CUDA backend
│   ├── libggml-cpu.so.*     # CPU backend
│   ├── libcurl.so.*         # HTTP client
│   ├── libssl.so.*          # SSL/TLS
│   ├── libcudart.so.*       # CUDA runtime (bundled)
│   ├── libnvrtc.so.*        # CUDA nvrtc (bundled)
│   ├── libcublas.so.*       # cuBLAS (bundled)
│   ├── libcublasLt.so.*     # cuBLASLt (bundled)
│   ├── libcusparse.so.*     # cuSPARSE (bundled)
│   ├── libnvjpeg.so.*       # NVIDIA JPEG (bundled)
│   ├── libgomp.so.*         # OpenMP runtime (bundled)
│   ├── libstdc++.so.*       # C++ standard library (bundled)
│   └── libgcc_s.so.*        # GCC runtime (bundled)
└── share/
    └── llama/               # Documentation and man pages
```

### Bundled Libraries

#### CUDA Runtime Libraries
- `libcudart.so.*` - CUDA Runtime API
- `libnvrtc.so.*` - CUDA NVRTC (NVCC Runtime Compilation)
- `libcurand.so.*` - CUDA Random Number Generation
- `libcublas.so.*` - CUDA Basic Linear Algebra Subprograms
- `libcublasLt.so.*` - CUDA cuBLAS Library
- `libcusparse.so.*` - CUDA Sparse Linear Algebra
- `libnvjpeg.so.*` - NVIDIA JPEG Library

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
/opt/vai-llama/cuda/bin/llama.cuda -m /path/to/model.gguf -c

# Server mode
/opt/vai-llama/cuda/bin/llama-server.cuda -m /path/to/model.gguf -c 2048

# Benchmark
/opt/vai-llama/cuda/bin/llama-bench.cuda -m /path/to/model.gguf -ngl 99
```

### REST API Server

```bash
# Start server
/opt/vai-llama/cuda/bin/llama-server.cuda -m /path/to/model.gguf -c 2048 -t 8

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
-DCMAKE_INSTALL_PREFIX=/opt/vai-llama/cuda
-DCMAKE_INSTALL_RPATH="\$ORIGIN/../lib;\$ORIGIN"
-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
-DGGML_NATIVE=OFF
-DGGML_VULKAN=OFF
-DGGML_CUDA=ON
-DGGML_HIP=OFF
-DGGML_METAL=OFF
-DGGML_BACKEND_DL=ON
-DGGML_CPU_ALL_VARIANTS=ON
-DLLAMA_BUILD_TESTS=OFF
-DLLAMA_BUILD_EXAMPLES=OFF
-DLLAMA_BUILD_TOOLS=ON
-DLLAMA_BUILD_SERVER=ON
-DLLAMA_OPENSSL=ON
-DLLAMA_BIN_NAME_SUFFIX=cuda
```

### GPU Backend Selection

The package is built with **CUDA only**:
- `GGML_CUDA=ON` - Enable CUDA backend
- `GGML_VULKAN=OFF` - Disable Vulkan
- `GGML_HIP=OFF` - Disable HIP/ROCm
- `GGML_METAL=OFF` - Disable Metal

---

## Troubleshooting

### Missing Libraries

If you encounter "library not found" errors:

```bash
# Check for missing libraries
ldd /opt/vai-llama/cuda/bin/llama.cuda

# Missing libraries should be bundled
# If still missing, check the package contents
dpkg -L vai-llama-cuda | grep lib
```

### GPU Detection

```bash
# Check NVIDIA GPUs
nvidia-smi

# Check CUDA version
nvcc --version
```

### Performance Issues

```bash
# Check CUDA device properties
nvidia-smi -L

# Check CUDA memory
nvidia-smi --query-compute-apps=summary --format=csv
```

---

## Package Size

Typical package sizes:

| Component | Size |
|-----------|------|
| Binaries | ~15 MB |
| CUDA Libraries | ~150 MB |
| System Libraries | ~50 MB |
| **Total** | **~215 MB** |

---

## License

This package is built from the llama-cpp-turboquant project. See the project's LICENSE file for details.

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/your-org/llama-cpp-turboquant/issues
- Documentation: https://github.com/your-org/llama-cpp-turboquant/docs
