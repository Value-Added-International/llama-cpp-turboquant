# Active Context

## Current Focus

Onboarded into MemPalace on 2026-09-27 by zoocode-tiel-coder-35b-A3B-MTP-262144-ctx. Wing `vai_llama_cpp_turboquant`
created and mined; memory-bank/ established; baseline manifest snapshot + handoff event recorded.

## Recent Changes

- AGENTS.md and `.roo/rules-code/AGENTS.md` corrected: ROCm/HIP preset `x64-linux-hip-rocm713-gfx1103-release` →
  `x64-linux-hip-rocm10-gfx1103-release` (matches `CMakePresets.json`; `rocm713` was stale).
- `.roo/rules-debug/AGENTS.md` trimmed three unverifiable claims (`--no-mmap`, `turbo3_tcq`/turbo4 sink-token,
  `--no-flash-attn`); kept verified HIP-graph and AOCC claims.
- `.roo/rules-code/AGENTS.md` gained the `test-quant-type-selection.cpp --generate` snapshot gotcha.
- Excluded `staging-*` (~1.7GB) from git tracking: `git rm -r --cached` + `.gitignore` commit `02bb9792a`.
  Files kept on disk; `.devops/build-*.sh` wipe + `cmake --install` them on every build.
- Renamed git branch `ROCM_7.13_AOCC` → `rocm10_aocc` (local + remote; old remote ref deleted, remote commit
  `16a414d68`). Updated the sole tracked doc reference in `vai-bench-mark-scripts/vai-llama-bench-plan.md`.

## Next Steps

- (Resolved) Branch label — now `rocm10_aocc`, consistent with the `rocm10` presets/Makefile.
- (Resolved) staging-* — excluded from git; no commit needed.

## Open Questions

- Resolved: branch label `ROCM_7.13_AOCC` was inconsistent with `rocm10` presets; renamed to `rocm10_aocc`.
  Note: `7.13` still refers to a real dependency (rocWMMA at the ROCm `rocwmma-rocm7.13` dir and
  `rocm/core-7.13/include`) while `rocm10` = HIP core-10.0 runtime — the two names describe different
  ROCm components, both real.

## Palace references

PALACE-REF | HANDOFF | vai_llama_cpp_turboquant/handoff | drawer (see palaceMap.md) | 2026-09-27 | zoocode-tiel-coder-35b-A3B-MTP-262144-ctx | onboarding handoff state
PALACE-REF | CHANGELOG | vai_llama_cpp_turboquant/changelog | drawer (see palaceMap.md) | 2026-09-27 | zoocode-tiel-coder-35b-A3B-MTP-262144-ctx | changelog of onboarding + AGENTS.md corrections
