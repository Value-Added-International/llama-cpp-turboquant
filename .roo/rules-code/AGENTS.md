# Project Coding Rules (Non-Obvious Only)

- C++ tests use custom framework from [`tests/testing.h`](tests/testing.h), not Google Test or Catch2
- Test files in `tests/` are compiled via `llama_build_and_test()` in [`tests/CMakeLists.txt`](tests/CMakeLists.txt)
- Test executables output to `build/bin/test-<name>` (e.g., `build/bin/test-arg-parser`)
- Filter tests by passing regex argument: `./build/bin/test-arg-parser "arg.*parser"`
- CMake presets in [`CMakePresets.json`](CMakePresets.json) define GPU backend configurations
- Build shortcuts via [`Makefile`](Makefile): `make hip-rocm713-gfx1103`, `make vulkan-aocc`, `make deb`
- Python type checking: `mypy` (strict mode in [`mypy.ini`](mypy.ini)), `pyright` (in [`pyrightconfig.json`](pyrightconfig.json))
- Python linting: `flake8` with `flake8-no-print` plugin (configured in [`.flake8`](.flake8) and [`.pre-commit-config.yaml`](.pre-commit-config.yaml))
- Pre-commit hooks: trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files, flake8
