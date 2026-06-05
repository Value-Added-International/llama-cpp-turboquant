NPROC   := $(shell nproc 2>/dev/null || echo 4)

BUILD_HIP    := build-x64-linux-hip-rocm713-gfx1103-release
BUILD_VULKAN := build-x64-linux-vulkan-aocc-release

.PHONY: help hip-rocm713-gfx1103 vulkan-aocc

.DEFAULT_GOAL := help

help:
	@printf "\nBuild system is CMake. Direct cmake usage:\n"
	@printf "  cmake -B build && cmake --build build -j$(NPROC)\n\n"
	@printf "Shortcut targets defined here:\n"
	@printf "  make hip-rocm713-gfx1103  -- HIP/ROCM 7.13 + AMD AOCC, gfx1103 (RDNA3)\n"
	@printf "  make vulkan-aocc          -- Vulkan + AMD AOCC (Zen4 CPU optimizations)\n\n"
	@printf "For full build docs: docs/build.md\n\n"

hip-rocm713-gfx1103:
	rm -rf $(BUILD_HIP)
	cmake --preset x64-linux-hip-rocm713-gfx1103-release
	cmake --build $(BUILD_HIP) -j$(NPROC)

vulkan-aocc:
	rm -rf $(BUILD_VULKAN)
	cmake --preset x64-linux-vulkan-aocc-release
	cmake --build $(BUILD_VULKAN) -j$(NPROC)
