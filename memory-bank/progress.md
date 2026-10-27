# Progress

## What Works

- Full llama.cpp fork with TurboQuant+ KV-cache and weight quantization across Metal, CUDA, ROCm, Vulkan, and CPU.
- Custom C++ test framework; per-model snapshot schema files in `tests/snapshots/`.
- Four packaged backend artifacts (CPU/CUDA/ROCM/VULKAN `.deb`) plus `staging-*` install trees.

## What Is Left / Status

- Not upstreamed; running as a long-lived feature branch (`feature/turboquant-kv-cache`, ~300 commits ahead of upstream
  per README; current checked-out branch is `ROCM_7.13_AOCC`).
- WIP per README status badge.

## Known Issues

- `test-quant-type-selection.cpp` requires network access (downloads remote models); fails offline.
- Branch label `ROCM_7.13_AOCC` vs ROCm 10.0 preset naming — needs reconciliation (see activeContext.md).

## Palace references

PALACE-REF | ISSUE | vai_llama_cpp_turboquant/issues | source tests/test-quant-type-selection.cpp | 2026-09-27 | zoocode-tiel-coder-35b-A3B-MTP-262144-ctx | test needs network / offline failure
PALACE-REF | HANDOFF | vai_llama_cpp_turboquant/handoff | drawer (see palaceMap.md) | 2026-09-27 | zoocode-tiel-coder-35b-A3B-MTP-262144-ctx | onboarding complete, verification pending
