# AMD ROCM 7.13 + AOCC 5.2.0 toolchain — HIP GPU (gfx1103 RDNA3) + Zen4 CPU
#
# CPU side (tokenizer, sampler, post-processing): AMD AOCC 5.2.0
# GPU side (HIP kernels): ROCM bundled amdclang — NOT system clang
#
# Override paths via env vars:
#   ROCM_PATH   (default: /opt/rocm)
#   AOCC_PATH   (default: /opt/AMD/aocc-compiler-5.2.0)

if(DEFINED ENV{ROCM_PATH})
    set(_rocm "$ENV{ROCM_PATH}")
else()
    set(_rocm "/opt/rocm")
endif()

if(DEFINED ENV{AOCC_PATH})
    set(_aocc "$ENV{AOCC_PATH}")
else()
    set(_aocc "/opt/AMD/aocc-compiler-5.2.0")
endif()

set(CMAKE_C_COMPILER   "${_aocc}/bin/clang"      CACHE FILEPATH "C compiler (AMD AOCC)")
set(CMAKE_CXX_COMPILER "${_aocc}/bin/clang++"    CACHE FILEPATH "C++ compiler (AMD AOCC)")
set(CMAKE_HIP_COMPILER "${_rocm}/llvm/bin/clang" CACHE FILEPATH "HIP compiler (ROCM amdclang)")

# Ensure ROCM CMake find_package modules locate headers and libraries
list(APPEND CMAKE_PREFIX_PATH "${_rocm}")

# HIP language enable (enable_language(HIP) in ggml-hip) reads these env vars
set(ENV{HIP_PATH} "${_rocm}")
set(ENV{HIPCXX}   "${_rocm}/llvm/bin/clang")
