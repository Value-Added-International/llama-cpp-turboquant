# AGENTS.md

This file provides guidance to agents when working with code in this repository.

> [!IMPORTANT]
> This project does **not** accept pull requests that are fully or predominantly AI-generated. AI tools may be utilized solely in an assistive capacity. Read [CONTRIBUTING.md](CONTRIBUTING.md) for full policy.

## Build Commands

**Standard build (Ninja generator):**
```bash
cmake -B build && cmake --build build -j$(nproc)
```

**Using CMake presets (recommended for GPU backends):**
```bash
cmake --preset x64-linux-hip-rocm10-gfx1103-release && cmake --build build-x64-linux-hip-rocm10-gfx1103-release -j$(nproc)
cmake --preset x64-linux-vulkan-aocc-release && cmake --build build-x64-linux-vulkan-aocc-release -j$(nproc)
```

**Makefile shortcuts:** `make hip-rocm10-gfx1103`, `make vulkan-aocc`, `make deb`

**Build specific target:** `cmake --build build --target server`

**Debug build:** `cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build`

## Test Commands

**Run all tests:**
```bash
cd build && ctest
```

**Run all tests verbose:**
```bash
cd build && ctest --verbose
```

**Run specific test binary:** `./build/bin/test-<name>` (e.g., `./build/bin/test-arg-parser`)

Tests use a custom C++ framework from [`tests/testing.h`](tests/testing.h) — not Google Test or Catch2. Test files are in `tests/` directory. Filter tests by name with command-line regex argument.

**Python tests:** `pytest` in relevant subdirectories (e.g., `tools/server/tests/`)

## Code Style

**C/C++ formatting:** `.clang-format` — 4-space indent, 120 column limit, `BreakBeforeBraces: Custom` (Attach), namespace braces NOT indented, break before multiline strings, break string literals, break constructor initializers after colon, break inheritance list after comma.

**C/C++ linting:** `.clang-tidy` — checks: `bugprone-*`, `readability-*`, `clang-analyzer-*`, `performance-*`, `portability-*`, `misc-*` (with specific exclusions listed in file).

**Python linting:** `.flake8` (with `flake8-no-print`), `mypy.ini` (strict mode), `pyrightconfig.json` (python 3.9+, with execution environment overrides).

**General:** `.editorconfig` — LF line endings, UTF-8, 4-space indent (tabs for Makefiles).

## Project Structure

- [`src/`](src/), [`include/`](include/) — Core llama library
- [`ggml/`](ggml/) — GGML tensor computation layer (CUDA, HIP, Metal, Vulkan, CPU backends)
- [`common/`](common/) — Shared utilities (sampling, tokenization, Jinja2 engine)
- [`tools/`](tools/) — llama-server, llama-bench, llama-quantize, etc.
- [`tests/`](tests/) — C++ unit tests (custom framework)
- [`examples/`](examples/) — Example programs

## TurboQuant Extensions (Fork-Specific)

This is a fork of llama.cpp with TurboQuant+ codec support. Key additions:
- Quantization types: `TQ3_1S`, `TQ4_1S` (weights), `turbo2`, `turbo3`, `turbo4` (KV cache)
- Default branch: `feature/turboquant-kv-cache` (long-lived feature branch, ~300 commits ahead of upstream)
- All existing llama.cpp features work unchanged — TurboQuant types are additive.

## ROCm/HIP Build Gotchas

- Requires `amdclang` from ROCm toolchain (not system clang) for GPU kernels
- Set `HIPCXX` and `HIP_PATH` via `hipconfig` before cmake
- For gfx1103: `GGML_HIP_GRAPHS` MUST be OFF (rocWMMA FA ops cannot be recorded in HIP graph stream)
- AOCC compiler for CPU-side code: `export PATH=/opt/AMD/aocc-compiler-5.2.0/bin:$PATH`

## Resources

- [CONTRIBUTING.md](CONTRIBUTING.md) — Contributor guidelines
- [docs/build.md](docs/build.md) — Comprehensive build guide
- [docs/development/HOWTO-add-model.md](docs/development/HOWTO-add-model.md) — Adding new models
- [tools/server/README.md](tools/server/README.md) — Server documentation
- [tools/server/README-dev.md](tools/server/README-dev.md) — Server development
- [docs/development/parsing.md](docs/development/parsing.md) — PEG parser
- [docs/autoparser.md](docs/autoparser.md) — Auto parser (PEG-based)
- [common/jinja/README.md](common/jinja/README.md) — Jinja engine
- [PR template](.github/pull_request_template.md)
