# Tech Context

## Technologies, Versions, Setup, Constraints, Dependencies

- Language/runtime: C++ (llama.cpp core) + Python (HF→GGUF converters, `gguf-py`, server tests).
- Build: CMake 4.x + Ninja; per-backend CMake presets; optional Makefile shortcuts.
- Backends: Metal (Apple Silicon), CUDA (NVIDIA), HIP/ROCm (AMD RDNA3/4 + CDNA3/4), Vulkan, SYCL, OpenCL, Hexagon, CPU.
- Package managers: pip (`requirements.txt`), Nix (`flake.nix`).
- Linters/type-checkers: `mypy` (strict, `mypy.ini`), `pyright` (`pyrightconfig.json`, Python 3.9+),
  `flake8` (`.flake8` + `flake8-no-print`), `ty` (`ty.toml`), `.clang-format`/`.clang-tidy` for C/C++.
- Tests: custom C++ framework (`tests/testing.h`); filter by regex CLI arg; `pytest` for Python.

## Constraints

- ROCm kernels require the ROCm toolchain (`amdclang`) and `hipconfig`-exported `HIPCXX`/`HIP_PATH`.
- gfx1103: `GGML_HIP_GRAPHS` must be OFF.
- TurboQuant types are additive; existing behaviour is unchanged.

## Shared Systems

Not found in project files — this project has no network/infra dependencies of its own; codec papers live in the
separate `TheTom/turboquant_plus` repo.

## Palace references

PALACE-REF | IDENTITY | vai_llama_cpp_turboquant/identity | source README.md | 2026-09-27 | zoocode-tiel-coder-35b-A3B-MTP-262144-ctx | stack, build, and test tooling
