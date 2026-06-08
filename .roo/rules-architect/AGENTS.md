# Project Architecture Rules (Non-Obvious Only)

- This is a fork of llama.cpp with TurboQuant+ codec support on `feature/turboquant-kv-cache` branch (~300 commits ahead of upstream)
- TurboQuant types are additive: all existing llama.cpp features work unchanged
- Quantization types: `TQ3_1S`, `TQ4_1S` (weights), `turbo2`, `turbo3`, `turbo4` (KV cache)
- GGML tensor layer (`ggml/`) is separate from llama library (`src/`, `include/`)
- Backend implementations: CUDA, HIP, Metal, Vulkan, CPU — each in `ggml/` subdirectories
- Common utilities (`common/`) shared across tools: sampling, tokenization, Jinja2 engine
- PR template: [`.github/pull_request_template.md`](.github/pull_request_template.md)
