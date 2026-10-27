# Project Brief

## Requirements, Goals, Scope

A fork of [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) (hosted at
[`Value-Added-International/llama-cpp-turboquant`](https://github.com/Value-Added-International/llama-cpp-turboquant))
that integrates the **TurboQuant+** codec stack for production-grade KV-cache and weight quantization.

Core goals:
- Deliver cross-backend kernel support (Apple Silicon Metal, NVIDIA CUDA, AMD ROCm/HIP, Vulkan) for quantized KV cache and weights.
- Provide additive, opt-in quant types that leave existing llama.cpp behaviour unchanged.
- Ship packaged `.deb`/`.so` artifacts for four backends (CPU, CUDA, ROCm, Vulkan).

Scope: the runtime integration only. Codec design, calibration, and validation papers live in the sibling repo
[`TheTom/turboquant_plus`](https://github.com/TheTom/turboquant_plus); this repo is the llama.cpp build/runtime.

## Palace references

PALACE-REF | IDENTITY | vai_llama_cpp_turboquant/identity | source README.md | 2026-09-27 | zoocode-tiel-coder-35b-A3B-MTP-262144-ctx | project identity and scope
PALACE-REF | PATTERN | vai_llama_cpp_turboquant/patterns | source AGENTS.md | 2026-09-27 | zoocode-tiel-coder-35b-A3B-MTP-262144-ctx | quant-type-selection snapshot test convention
PALACE-REF | PATTERN | vai_llama_cpp_turboquant/patterns | source tests/test-quant-type-selection.cpp | 2026-09-27 | zoocode-tiel-coder-35b-A3B-MTP-262144-ctx | snapshot regeneration via --generate
