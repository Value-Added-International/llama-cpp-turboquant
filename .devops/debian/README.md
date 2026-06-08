# VAI Llama-cpp-turboquant Packaging System

## Overview

This directory contains the packaging infrastructure for creating self-contained Debian packages of llama-cpp-turboquant with GPU acceleration. Each package includes all runtime dependencies, allowing installation to `/opt/vai-llama` without requiring system-wide dependency installation.

## Package Structure

```
.devops/debian/
├── vulkan/                   # Vulkan-only package
│   ├── control              # Debian control file (package metadata)
│   └── postinst             # Post-installation script
├── cuda/                     # NVIDIA CUDA package
│   ├── control              # Debian control file (package metadata)
│   └── postinst             # Post-installation script
└── README.md                 # This documentation
```

## Build Scripts

### Vulkan Package Build Script
**Location:** `.devops/build-vulkan-package.sh`

This script builds a self-contained Vulkan-accelerated package containing:
- `llama-cli` - Command-line LLM inference tool
- `llama-server` - REST API server for LLM serving
- `llama-bench` - Performance benchmarking tool
- All required runtime libraries bundled in the package

**Prerequisites:**
- CMake 3.20+ with Vulkan support
- Debian/Ubuntu build environment (build-essential, libssl-dev)
- Vulkan SDK or compatible headers/libraries for cross-compilation

### CUDA Package Build Script
**Location:** `.devops/build-cuda-package.sh`

This script builds a self-contained NVIDIA CUDA-accelerated package containing:
- `llama-cli` - Command-line LLM inference tool
- `llama-server` - REST API server for LLM serving
- `llama-bench` - Performance benchmarking tool
- All required runtime libraries bundled in the package

**Prerequisites:**
- CMake 3.20+ with CUDA support (CUDA toolkit or compatible stubs)
- Debian/Ubuntu build environment (build-essential, libssl-dev)
- CUDA headers/libraries for cross-compilation

## Installation Layout

Both packages install to `/opt/vai-llama` with the following structure:

```
/opt/vai-llama/
├── bin/                    # Executable binaries
│   ├── llama-cli          # Command-line client
│   ├── llama-server       # REST API server
│   └── llama-bench        # Benchmarking tool
├── lib/                    # Shared libraries (bundled)
│   ├── libllama.so.*      # Main library with GPU backend
│   ├── libggml-vulkan.so.*  # Vulkan compute backend
│   └── libggml-cuda.so.*    # CUDA compute backend (CUDA package only)
├── share/
│   └── doc/                # Documentation and man pages
└── include/                # Header files for development
```

## Usage

### Building Packages

From the project root directory:

```bash
# Build Vulkan package
cd .devops/debian/vulkan && bash ../build-vulkan-package.sh

# Build CUDA package  
cd .devops/debian/cuda && bash ../build-cuda-package.sh
```

Both scripts output `.deb` packages to their respective directories.

### Installing Packages (Manual)

After building, manually install the packages:

```bash
# Install Vulkan package
dpkg -i vai-llama-vulkan_0.0.*_amd64.deb

# Install CUDA package
dpkg -i vai-llama-cuda_0.0.*_amd64.deb
```

### Running After Installation

The binaries are available at:

```bash
/opt/vai-llama/bin/llama-cli --help
/opt/vai-llama/bin/llama-server --help
/opt/vai-llama/bin/llama-bench --help
```

## Design Rationale

### Why Self-Contained Packages?

1. **Zero Runtime Dependencies**: Systems only need the GPU drivers (Vulkan or CUDA) - no additional packages required
2. **Consistent Deployment**: Identical behavior across different target systems
3. **Simplified Maintenance**: No dependency conflicts with system libraries
4. **Easy Distribution**: Single package per backend type

### Why Two Separate Packages?

1. **Hardware Specificity**: Vulkan and CUDA are mutually exclusive GPU backends
2. **Reduced Package Size**: Each package is smaller, containing only relevant dependencies
3. **Flexibility**: Systems can choose the appropriate backend based on hardware
4. **Conflict Avoidance**: Prevents library conflicts between different GPU backends

### Bundling Strategy

All runtime libraries are bundled within each package:
- Shared object files use RPATH pointing to `$ORIGIN/../lib`
- OpenSSL and other dependencies are statically linked or copied
- The Debian control file declares minimal system requirements (libc6, libstdc++6)

## CMake Presets

### Vulkan Package Preset
**Location:** `.devops/cmake-preset-vulkan.json`

Configures the build with:
```cmake
GGML_VULKAN=ON
GGML_CUDA=OFF
CMAKE_INSTALL_RPATH=$ORIGIN/../lib;$ORIGIN
```

### CUDA Package Preset  
**Location:** `.devops/cmake-preset-cuda.json`

Configures the build with:
```cmake
GGML_CUDA=ON
GGML_VULKAN=OFF
CMAKE_INSTALL_RPATH=$ORIGIN/../lib:$ORIGIN
```

## Versioning

Package versions follow the format `0.0.$COMMIT_COUNT`, where COMMIT_COUNT is the number of commits in the repository. This ensures unique version numbers for each build while maintaining semantic versioning principles.

## License

This packaging infrastructure follows the MIT license, consistent with the llama-cpp-turboquant project.
