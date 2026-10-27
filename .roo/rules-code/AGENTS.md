# Project Coding Rules (Non-Obvious Only)

- C++ tests use custom framework from [`tests/testing.h`](tests/testing.h), not Google Test or Catch2
- Test files in `tests/` are compiled via `llama_build_and_test()` in [`tests/CMakeLists.txt`](tests/CMakeLists.txt)
- Test executables output to `build/bin/test-<name>` (e.g., `build/bin/test-arg-parser`)
- Filter tests by passing regex argument: `./build/bin/test-arg-parser "arg.*parser"`
- CMake presets in [`CMakePresets.json`](CMakePresets.json) define GPU backend configs; ROCm/HIP preset is `x64-linux-hip-rocm10-gfx1103-release` (not `rocm713`)
- Build shortcuts via [`Makefile`](Makefile): `make hip-rocm10-gfx1103`, `make vulkan-aocc`, `make deb`
- Python type checking: `mypy` (strict mode in [`mypy.ini`](mypy.ini)), `pyright` (in [`pyrightconfig.json`](pyrightconfig.json))
- Python linting: `flake8` with `flake8-no-print` plugin (configured in [`.flake8`](.flake8) and [`.pre-commit-config.yaml`](.pre-commit-config.yaml))
- Pre-commit hooks: trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files, flake8
- `tests/test-quant-type-selection.cpp` compares quant output against `tests/snapshots/*.schema`; run with `--generate` to regenerate them after changing quant logic (not via ctest)
- Quantization types are fork-specific: `TQ3_1S`/`TQ4_1S` (weights), `turbo2`/`turbo3`/`turbo4` (KV cache); additive, existing llama.cpp behavior unchanged
- `tests/test-quant-type-selection.cpp` links `llama`/`llama-common` and needs remote models (HuggingFace) — it will fail offline without network access
