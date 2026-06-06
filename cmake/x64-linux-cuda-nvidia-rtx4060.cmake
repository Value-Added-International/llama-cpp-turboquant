# NVIDIA RTX 4060 Laptop (Ada Lovelace, sm_89) — CUDA backend
#
# GPU: NVIDIA CUDA, compiled by nvcc from CUDA toolkit.
# CPU: system GCC/clang (no AOCC; AOCC is AMD-only).
#
# Override paths via env vars:
#   CUDA_PATH   (default: /usr/local/cuda)

if(DEFINED ENV{CUDA_PATH})
    set(_cuda "$ENV{CUDA_PATH}")
else()
    set(_cuda "/usr/local/cuda")
endif()

set(CMAKE_CUDA_COMPILER "${_cuda}/bin/nvcc" CACHE FILEPATH "CUDA compiler (nvcc)")

set(GGML_CUDA        ON  CACHE BOOL "" FORCE)
set(GGML_HIP         OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN      OFF CACHE BOOL "" FORCE)
set(GGML_METAL       OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE      ON  CACHE BOOL "x86_64 native instructions (AVX2/AVX512)" FORCE)

# Ada Lovelace = sm_89; also support sm_86 (Ampere RTX 30xx) for portability.
# Adjust CMAKE_CUDA_ARCHITECTURES if targeting a different GPU.
set(CMAKE_CUDA_ARCHITECTURES "89;86" CACHE STRING "CUDA architectures" FORCE)

# Flash attention for all quantised KV types (q8_0, turbo3, turbo4, etc.)
set(GGML_CUDA_FA_ALL_QUANTS ON CACHE BOOL "" FORCE)

set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
