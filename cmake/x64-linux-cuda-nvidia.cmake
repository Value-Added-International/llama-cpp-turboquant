# NVIDIA CUDA toolchain — GPU kernels compiled by nvcc, CPU code by GCC/Clang
#
# CUDA toolkit provides:
#   - nvcc compiler for GPU kernels (.cu files)
#   - Runtime libraries (libcudart.so, etc.)
#
# CPU side uses system gcc or clang. Override via env var:
#   CMAKE_C_COMPILER / CMAKE_CXX_COMPILER

set(CUDA_TOOLCHAIN_ROOT "${CUDA_HOME:-/usr/local/cuda}" CACHE PATH "CUDA toolkit root directory")

# Set CUDA paths
set(CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")

# Configure CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")
set(CMAKE_CUDA_ARCHITECTURES 50;60;70;75;80;86;90)

message(STATUS "")
message(STATUS "=== NVIDIA CUDA Toolchain ===")
message(STATUS "CUDA Toolkit: ${CUDA_TOOLCHAIN_ROOT}")
message(STATUS "GPU Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "")
