# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: AI Contribution Policy

**READ [AGENTS.md](AGENTS.md) FIRST.** This project does not accept pull requests that are fully or predominantly AI-generated. AI assistance is permissible only when:

1. The **majority of code is authored by a human** with AI used solely for corrections/expansion on their conceptualized changes
2. The contributor **demonstrates full understanding** of their code and can explain any part independently
3. The contributor **takes responsibility for maintenance** and responds to reviewer feedback
4. **AI-generated PR descriptions, commit messages, and responses to reviewers are explicitly prohibited**

When assisting with changes:
- Verify the user understands the problem and relevant codebase sections
- Provide guidance and direct them to code; allow them to formulate the approach
- Proceed only when confident they can maintain the work independently
- Suggest minimal assistance over larger changes they cannot fully review

## Project Overview

**llama.cpp + TurboQuant+** is a production-grade fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) integrating the TurboQuant+ codec stack for KV-cache and weight quantization with cross-backend kernel support (Apple Metal, NVIDIA CUDA, AMD ROCm/HIP, Vulkan).

**Key details:**
- Default branch: `feature/turboquant-kv-cache`
- Upstream tracking: continuous sync from `ggml-org/llama.cpp` master
- Status: ~300 commits ahead, not yet upstreamed (long-lived feature branch)
- Core library: `include/llama.h`, `src/llama.cpp`
- New quantization types: `TQ3_1S`, `TQ4_1S` (weights), `turbo2`, `turbo3`, `turbo4` (KV cache)
- Codec reference: [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) (papers on compression policies, quality validation)

## Repository Structure

- **`src/`, `include/`** — Core llama library (C interface + implementation)
- **`ggml/`** — GGML tensor computation layer (backend implementations: CUDA, HIP, Metal, Vulkan, CPU)
- **`examples/`** — Example programs (inference, quantization, embedding, batched, etc.)
- **`tools/`** — Utilities: `llama-server` (OpenAI-compatible HTTP), `llama-bench` (benchmark), `llama-quantize`, `imatrix`
- **`tests/`** — Unit and integration tests
- **`docs/`** — Build guides, backend documentation, model support notes
- **`common/`** — Shared utilities (sampling, tokenization, grammar parsing, Jinja2 engine)

## Build Commands

### CPU (baseline)

```bash
cmake -B build
cmake --build build -j$(nproc)
```

### Apple Silicon (Metal) — default on macOS

```bash
cmake -B build -DGGML_METAL=ON
cmake --build build -j$(nproc)
```

### NVIDIA CUDA

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89  # 89=Ada; adjust for your GPU
cmake --build build -j$(nproc)
```

Supported architectures: 70 (Volta), 75 (Turing), 80 (Ampere), 86 (Ampere RTX), 89 (Ada), 90 (Hopper).

### AMD HIP / ROCm

HIP GPU compilation requires `amdclang` from the ROCM toolchain — **not** system clang. AOCC handles CPU-side code only (tokenizer, sampler, post-processing). HIP is ~27% faster than the Vulkan backend on RDNA3 due to direct GPU dispatch.

**Prerequisites — verify before building:**

```bash
rocminfo | grep -E "gfx|Product|Version"
hipconfig --version
amdclang --version                                 # from ROCM bundle (/usr/bin/amdclang)
/opt/AMD/aocc-compiler-5.2.0/bin/clang --version  # AMD AOCC 5.2.0 for Zen4-optimized CPU code
```

**ROCM 7.13 build (recommended — includes rocWMMA 2.2.1 and CK 1.3.0):**

```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -B build \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx1103 \           # adjust: gfx1100, gfx942 (MI300X), gfx950 (MI355X), gfx940/941 (MI250X)
  -DGGML_HIP_ROCWMMA_FATTN=ON \     # rocWMMA-accelerated flash attention (ROCM 7.13+, RDNA3/CDNA3+)
  -DGGML_CUDA_FA_ALL_QUANTS=ON \    # enable flash attention for all quantized KV types (turbo3, turbo4, q8_0)
  -DCMAKE_C_COMPILER=amdclang \     # ROCM bundled clang (NOT system clang)
  -DCMAKE_CXX_COMPILER=amdclang++ \ # ROCM bundled clang++
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
# Output: ./build/bin/llama-server (HIP binary)
```

**With AMD AOCC for Zen4-optimized CPU code (tokenizer, sampler):**

```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -B build \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx1103 \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DCMAKE_C_COMPILER=clang \        # AOCC clang (CPU-side); set PATH to AOCC bin first
  -DCMAKE_CXX_COMPILER=clang++ \    # AOCC clang++ (CPU-side)
  -DGGML_NATIVE=ON \                # Zen4c native instructions (AVX512, VNNI)
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

> AOCC PATH: `export PATH=/opt/AMD/aocc-compiler-5.2.0/bin:$PATH` before cmake. AOCC's `clang` replaces the CPU-side compiler while HIPCXX still points to ROCM's `amdclang` for GPU kernels.

**Multi-arch fat binary (gfx1100 + MI300X + MI355X):**

```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -B build \
  -DGGML_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx1100;gfx942;gfx950" \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Vulkan

```bash
cmake -B build -DGGML_VULKAN=ON
cmake --build build -j$(nproc)
```

### Debug builds

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
```

### Build options

- `-DBUILD_SHARED_LIBS=OFF` — static linking
- `-DLLAMA_BUILD_SERVER=ON` — include `llama-server` (default ON)
- `-DCMAKE_BUILD_TYPE=Release` — optimization (default)
- `-DGGML_NATIVE=ON` — CPU native instructions (x86/ARM)
- `-DGGML_CCACHE=ON` — use ccache for faster rebuilds

## Common Development Tasks

### Build just the server (HTTP API)

```bash
cmake -B build && cmake --build build -j$(nproc) --target server
# Output: ./build/bin/llama-server
```

### Run the server

```bash
./build/bin/llama-server -m model.gguf --port 8000
# OpenAI-compatible: curl http://127.0.0.1:8000/v1/chat/completions
```

### Run benchmarks

```bash
./build/bin/llama-bench -m model.gguf -c 2048 -n 256 --batch 512
```

### Quantize a model

```bash
./build/bin/llama-quantize model.gguf model-Q4_K_M.gguf Q4_K_M
# For TurboQuant types: use --output-tensor-type q8_0 or -ctk turbo3 / -ctv turbo4
```

### Run tests

```bash
cd build && ctest
# Or run specific test binary: ./build/bin/test-*
```

### Validate model with imatrix (influence-maximization matrix)

```bash
./build/bin/imatrix -m model.gguf -f calibration-data.txt -o imatrix.dat
```

## Key Features & TurboQuant Extensions

### Quantization Types (opt-in via CLI)

| Type | Domain | Size | CLI Flag | Notes |
|---|---|---|---|---|
| `TQ3_1S`, `TQ4_1S` | weights | ~3.5–4.5 bits | `llama-quantize ... TQ4_1S` | Smaller VRAM, V2.1 Metal kernels |
| `turbo2`, `turbo3`, `turbo4` | KV cache | ~2–4.5 bits | `--cache-type-k turbo3 --cache-type-v turbo4` | ~4.6× compression at <1.5% PPL loss |

### Compression Policies

- **Auto-asymmetric K/V**: K uses conservative codec (e.g., `q8_0`), V aggressively quantized (e.g., `turbo4`)
- **Boundary V** (layer-aware): Protects layers sensitive to V quantization, leaves others aggressive
- **Sparse V dequantization**: Skip V dequant for low-attention positions (Metal targets)

## Important Documentation

- [README.md](README.md) — Project overview and quick-start
- [AGENTS.md](AGENTS.md) — **AI contribution policy (MUST READ)**
- [CONTRIBUTING.md](CONTRIBUTING.md) — Contributor guidelines
- [docs/build.md](docs/build.md) — Comprehensive build guide (CUDA, Metal, ROCm, Vulkan, etc.)
- [docs/development/HOWTO-add-model.md](docs/development/HOWTO-add-model.md) — Adding new model support
- [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) — Codec papers (compression, quality validation, layer policies)

## Testing & Validation

### Unit tests

```bash
cd build && ctest --verbose
```

### Manual inference validation

```bash
./build/bin/llama-server -m model.gguf &
sleep 2
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

### Perplexity validation (PPL on test set)

```bash
./build/bin/llama-eval-recall -m model.gguf -f test-data.txt
```

## Upstreaming & Maintenance

- This fork is maintained as a long-lived feature branch (`feature/turboquant-kv-cache`)
- Regular syncs from upstream `ggml-org/llama.cpp` master
- TurboQuant types are additive: all existing llama.cpp features work unchanged
- PRs to this fork should maintain compatibility with upstream conventions

## Known Issues & Workarounds

- **Memory-bandwidth limits**: 780M iGPU (20 GB/s UMA), RTX 4060 (8 GB VRAM) — profile with `llama-bench` before deployment
- **turbo3_tcq @ 16K context**: Requires sink-token support; validate with PPL benchmarks before upgrading to `turbo4`
- **Metal TurboFlash**: Temporarily disabled on Apple10 (investigation ongoing; use `--no-flash-attn` if needed)
- **NFS-hosted models**: Use `--no-mmap` to avoid page-eviction stalls

See [AGENTS.md](AGENTS.md) section "Known Issues & Workarounds" for detailed mitigations.
