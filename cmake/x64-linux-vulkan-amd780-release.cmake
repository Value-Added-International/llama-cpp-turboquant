# AMD 780M (RDNA3 iGPU) — Vulkan backend with AOCC 5.2.0 Zen4 CPU code
#
# GPU shaders: compiled by glslc at build time (no separate GPU compiler needed).
# CPU side (tokenizer, sampler, post-processing): AMD AOCC 5.2.0.
#
# Override paths via env vars:
#   AOCC_PATH   (default: /opt/AMD/aocc-compiler-5.2.0)

if(DEFINED ENV{AOCC_PATH})
    set(_aocc "$ENV{AOCC_PATH}")
else()
    set(_aocc "/opt/AMD/aocc-compiler-5.2.0")
endif()

set(CMAKE_C_COMPILER   "${_aocc}/bin/clang"   CACHE FILEPATH "C compiler (AMD AOCC)")
set(CMAKE_CXX_COMPILER "${_aocc}/bin/clang++" CACHE FILEPATH "C++ compiler (AMD AOCC)")

set(GGML_VULKAN      ON  CACHE BOOL "" FORCE)
set(GGML_HIP         OFF CACHE BOOL "" FORCE)
set(GGML_CUDA        OFF CACHE BOOL "" FORCE)
set(GGML_METAL       OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE      ON  CACHE BOOL "Zen4 native instructions (AVX512, VNNI)" FORCE)
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
