# Build and Package Reference — gfx1103 HIP / ROCm 7.13

## Prerequisites

Verify the toolchain before building:

```bash
rocminfo | grep -E "gfx|Product|Version"
hipconfig --version
amdclang --version                                  # ROCM bundled — /usr/bin/amdclang
/opt/AMD/aocc-compiler-5.2.0/bin/clang --version   # AMD AOCC 5.2.0 for Zen4 CPU code
```

Required system packages:

```bash
sudo apt-get install cmake ninja-build fakeroot dpkg-dev python3
```

---

## Build

### Shortcut (recommended)

```bash
make hip-rocm713-gfx1103
```

This does a full clean configure + build into `build-x64-linux-hip-rocm713-gfx1103-release/`.

### Manual cmake

```bash
cmake --preset x64-linux-hip-rocm713-gfx1103-release
cmake --build build-x64-linux-hip-rocm713-gfx1103-release -j$(nproc)
```

### Incremental rebuild (after source changes only)

```bash
cmake --build build-x64-linux-hip-rocm713-gfx1103-release -j$(nproc)
```

No reconfigure needed unless `CMakeLists.txt` or `CMakePresets.json` changed.

### Key flags set by the preset

| Flag | Value | Reason |
|---|---|---|
| `GGML_HIP` | `ON` | AMD HIP/ROCm backend |
| `GPU_TARGETS` | `gfx1103` | Radeon 780M (RDNA3 iGPU) |
| `GGML_HIP_ROCWMMA_FATTN` | `ON` | rocWMMA flash attention (ROCm 7.13+) |
| `GGML_CUDA_FA_ALL_QUANTS` | `ON` | FA for turbo3 / turbo4 / q8_0 KV types |
| `GGML_HIP_GRAPHS` | `OFF` | rocWMMA FA ops cannot be recorded in a HIP graph stream on gfx1103 |
| `GGML_NATIVE` | `ON` | Zen4c AVX512 / VNNI CPU instructions |

---

## Smoke test

Run after every build to confirm the HIP backend and turbo KV cache are working:

```bash
# Phase 1 — HIP backend loads (baseline, no turbo)
./smoke-test.sh 1

# Phase 2 — turbo4 V cache + flash attention
./smoke-test.sh 2

# Phase 3 — recommended production config (q8_0 K + turbo3 V)
./smoke-test.sh 3
```

Expected decode throughput on the 780M (Gemma 4 2B Q8_0):

| Phase | Config | t/s |
|---|---|---|
| 1 | f16/f16 | ~10 |
| 2 | f16 K + turbo4 V | ~13 |
| 3 | q8_0 K + turbo3 V | ~17 |

---

## Deb package

### 1. Stage the install

```bash
VERSION=$(cat build-x64-linux-hip-rocm713-gfx1103-release/llama.pc \
          | grep '^Version:' | cut -d' ' -f2)
PKG=llama-cpp-turboquant-hip-gfx1103_${VERSION}_amd64

cmake --install build-x64-linux-hip-rocm713-gfx1103-release \
      --prefix ${PKG}/usr/local
```

### 2. Write the control file

```bash
mkdir -p ${PKG}/DEBIAN

cat > ${PKG}/DEBIAN/control <<EOF
Package: llama-cpp-turboquant-hip-gfx1103
Version: ${VERSION}
Architecture: amd64
Maintainer: Value-Added International <value.added.kr@gmail.com>
Depends: amdrocm-core7.13
Description: llama.cpp + TurboQuant+ — HIP/ROCm 7.13 build for gfx1103 (Radeon 780M)
 Production-grade KV-cache quantization (turbo2/3/4) and weight quantization
 (TQ3_1S, TQ4_1S) with rocWMMA flash attention for AMD RDNA3 iGPU targets.
EOF
```

### 3. Register shared libraries on install/remove

```bash
cat > ${PKG}/DEBIAN/postinst <<'EOF'
#!/bin/sh
ldconfig
EOF

cat > ${PKG}/DEBIAN/postrm <<'EOF'
#!/bin/sh
ldconfig
EOF

chmod 755 ${PKG}/DEBIAN/postinst ${PKG}/DEBIAN/postrm
```

### 4. Build the .deb

```bash
fakeroot dpkg-deb --build ${PKG}
```

Output: `llama-cpp-turboquant-hip-gfx1103_<version>_amd64.deb`

### 5. Install / verify

```bash
sudo dpkg -i ${PKG}.deb
which llama-server && llama-server --version
```

### Full one-liner sequence

```bash
VERSION=$(cat build-x64-linux-hip-rocm713-gfx1103-release/llama.pc \
          | grep '^Version:' | cut -d' ' -f2) && \
PKG=llama-cpp-turboquant-hip-gfx1103_${VERSION}_amd64 && \
cmake --install build-x64-linux-hip-rocm713-gfx1103-release --prefix ${PKG}/usr/local && \
mkdir -p ${PKG}/DEBIAN && \
printf "Package: llama-cpp-turboquant-hip-gfx1103\nVersion: ${VERSION}\nArchitecture: amd64\nMaintainer: Value-Added International <value.added.kr@gmail.com>\nDepends: amdrocm-core7.13\nDescription: llama.cpp + TurboQuant+ HIP/ROCm 7.13 gfx1103\n" > ${PKG}/DEBIAN/control && \
printf '#!/bin/sh\nldconfig\n' | tee ${PKG}/DEBIAN/postinst ${PKG}/DEBIAN/postrm > /dev/null && \
chmod 755 ${PKG}/DEBIAN/postinst ${PKG}/DEBIAN/postrm && \
fakeroot dpkg-deb --build ${PKG}
```

---

## What gets installed

| Path | Contents |
|---|---|
| `/usr/local/bin/llama-server` | OpenAI-compatible HTTP inference server |
| `/usr/local/bin/llama-cli` | Interactive CLI inference |
| `/usr/local/bin/llama-quantize` | Weight quantization (TQ4_1S, Q4_K_M, etc.) |
| `/usr/local/bin/llama-bench` | Throughput benchmarks |
| `/usr/local/bin/llama-imatrix` | Influence-matrix calibration |
| `/usr/local/bin/llama-perplexity` | PPL quality validation |
| `/usr/local/lib/libggml-hip.so.*` | HIP backend (gfx1103 kernels) |
| `/usr/local/lib/libggml*.so.*` | GGML compute library |
| `/usr/local/lib/libllama.so.*` | Core llama library |
| `/usr/local/include/llama.h` | Public C API |
| `/usr/local/lib/pkgconfig/llama.pc` | pkg-config integration |
