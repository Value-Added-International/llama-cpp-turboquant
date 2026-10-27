# Project Debug Rules (Non-Obvious Only)

- Debug build: `cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build`
- HIP graph recording fails on gfx1103: `GGML_HIP_GRAPHS` MUST be OFF or HIP graph streams break
- AOCC compiler for CPU-side code requires: `export PATH=/opt/AMD/aocc-compiler-5.2.0/bin:$PATH` before cmake
- Memory-bandwidth limited devices (780M iGPU: 20 GB/s UMA, RTX 4060: 8 GB VRAM) — profile with `llama-bench` before deployment
