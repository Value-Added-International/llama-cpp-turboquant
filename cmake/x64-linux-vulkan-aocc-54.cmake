# AMD AOCC 5.4.0 toolchain — Vulkan backend with Zen4-optimized CPU code
#
# GPU shaders compiled by glslc (separate from C++ compiler).
# CPU side (tokenizer, sampler, post-processing): AMD AOCC 5.4.0.
#
# Override path via env var:
#   AOCC_PATH   (default: /opt/AMD/aocc-compiler-5.4.0)

if(DEFINED ENV{AOCC_PATH})
    set(_aocc "$ENV{AOCC_PATH}")
else()
    set(_aocc "/opt/AMD/aocc-compiler-5.4.0")
endif()

set(CMAKE_C_COMPILER   "${_aocc}/bin/clang"   CACHE FILEPATH "C compiler (AMD AOCC 5.4)")
set(CMAKE_CXX_COMPILER "${_aocc}/bin/clang++" CACHE FILEPATH "C++ compiler (AMD AOCC 5.4)")

# Set AOCC-specific flags for better optimization with Vulkan workloads
if(DEFINED ENV{AOCC_PATH})
    set(_hip_cxxflags "$ENV{HIPCXX}")
else()
    # Default HIPCXX path for AOCC 5.4
    set(_hip_cxxflags "-I${_aocc}/lib/clang/17/include")
endif()

# Export environment variables needed by the compiler
set(CMAKE_C_FLAGS_INIT " -fPIC ")
set(CMAKE_CXX_FLAGS_INIT " -fPIC ${_hip_cxxflags}")

# Linker flags
if(DEFINED ENV{HIPCXX})
    set(_linker_flags "$ENV{HIPCXX} -L${_aocc}/lib64")
else()
    set(_linker_flags "-L${_aocc}/lib64")
endif()

set(CMAKE_EXE_LINKER_FLAGS_INIT " ${_linker_flags}")

message(STATUS "")
message(STATUS "=== AMD AOCC 5.4.0 Vulkan Toolchain ===")
message(STATUS "AOCC Path: ${_aocc}")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "HIPCXX: ${_hip_cxxflags}")
message(STATUS "")
