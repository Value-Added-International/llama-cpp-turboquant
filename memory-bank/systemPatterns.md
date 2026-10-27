# System Patterns

## Architecture, Decisions, Patterns, Component Relationships

Structure mirrors upstream llama.cpp:
- `ggml/` — tensor compute layer + per-backend kernels (CUDA, HIP/ROCm, Metal, Vulkan, CPU, SYCL, OpenCL, Hexagon, etc.).
- `src/`, `include/` — core llama library (`llama-quant.cpp`, `llama-model-loader.cpp`, `llama.cpp`).
- `common/` — shared utilities: sampling, tokenization, Jinja2 chat-template engine, PEG/grammar parsing.
- `tools/` — llama-server, llama-bench, llama-quantize.
- `conversion/` — Python HF→GGUF converters (one module per model family).
- `tests/` — C++ unit tests via a custom framework in `tests/testing.h` (not Google Test / Catch2).
- `staging-*` — per-backend packaging dirs (`.so`, `.deb`); excluded from the palace via `.gitignore`.

Key non-obvious conventions:
- Quant types are fork-specific and additive: `TQ3_1S`/`TQ4_1S` (weights), `turbo2`/`turbo3`/`turbo4` (KV cache).
- `test-quant-type-selection.cpp` compares quant output against `tests/snapshots/*.schema`; regenerate with
  `--generate` (not via `ctest`). The test links `llama`/`llama-common` and downloads remote models, so it fails offline.
- Build: CMake + Ninja, presets in `CMakePresets.json` per backend. ROCm/HIP preset is
  `x64-linux-hip-rocm10-gfx1103-release`; Vulkan preset `x64-linux-vulkan-aocc-release`. Makefile shortcuts:
  `make hip-rocm10-gfx1103`, `make vulkan-aocc`, `make deb`.
- ROCm gotchas: kernels need `amdclang` from the ROCm toolchain (not system clang); `GGML_HIP_GRAPHS` MUST be OFF for
  gfx1103 (rocWMMA FA ops cannot be recorded in a HIP graph stream); AOCC CPU build needs
  `export PATH=/opt/AMD/aocc-compiler-5.2.0/bin:$PATH`.

## Palace references

PALACE-REF | PATTERN | vai_llama_cpp_turboquant/patterns | source AGENTS.md | 2026-09-27 | zoocode-tiel-coder-35b-A3B-MTP-262144-ctx | architecture and quant/snapshot/build conventions
PALACE-REF | DECISION | vai_llama_cpp_turboquant/decisions | source AGENTS.md | 2026-09-27 | zoocode-tiel-coder-35b-A3B-MTP-262144-ctx | ROCm preset name corrected to rocm10 (verified vs CMakePresets.json)
