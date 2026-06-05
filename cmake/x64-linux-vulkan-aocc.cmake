# AMD AOCC 5.2.0 toolchain — Vulkan backend with Zen4-optimized CPU code
#
# GPU shaders compiled by glslc (separate from C++ compiler).
# CPU side (tokenizer, sampler, post-processing): AMD AOCC 5.2.0.
#
# Override path via env var:
#   AOCC_PATH   (default: /opt/AMD/aocc-compiler-5.2.0)

if(DEFINED ENV{AOCC_PATH})
    set(_aocc "$ENV{AOCC_PATH}")
else()
    set(_aocc "/opt/AMD/aocc-compiler-5.2.0")
endif()

set(CMAKE_C_COMPILER   "${_aocc}/bin/clang"   CACHE FILEPATH "C compiler (AMD AOCC)")
set(CMAKE_CXX_COMPILER "${_aocc}/bin/clang++" CACHE FILEPATH "C++ compiler (AMD AOCC)")
