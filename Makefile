NPROC   := $(shell nproc 2>/dev/null || echo 4)

BUILD_HIP    := build-x64-linux-hip-rocm713-gfx1103-release
BUILD_VULKAN := build-x64-linux-vulkan-aocc-release

VERSION := $(shell grep '^Version:' $(BUILD_HIP)/llama.pc 2>/dev/null | cut -d' ' -f2)
DEB_PKG := llama-cpp-turboquant-hip-gfx1103_$(VERSION)_amd64

.PHONY: help hip-rocm713-gfx1103 vulkan-aocc deb

.DEFAULT_GOAL := help

help:
	@printf "\nBuild system is CMake. Direct cmake usage:\n"
	@printf "  cmake -B build && cmake --build build -j$(NPROC)\n\n"
	@printf "Shortcut targets defined here:\n"
	@printf "  make hip-rocm713-gfx1103  -- HIP/ROCM 7.13 + AMD AOCC, gfx1103 (RDNA3)\n"
	@printf "  make vulkan-aocc          -- Vulkan + AMD AOCC (Zen4 CPU optimizations)\n"
	@printf "  make deb                  -- build .deb from $(BUILD_HIP) (requires prior build)\n\n"
	@printf "For full build and packaging docs: make-help.md\n\n"

hip-rocm713-gfx1103:
	rm -rf $(BUILD_HIP)
	cmake --preset x64-linux-hip-rocm713-gfx1103-release
	cmake --build $(BUILD_HIP) -j$(NPROC)

vulkan-aocc:
	rm -rf $(BUILD_VULKAN)
	cmake --preset x64-linux-vulkan-aocc-release
	cmake --build $(BUILD_VULKAN) -j$(NPROC)

deb: $(BUILD_HIP)/llama.pc
	rm -rf $(DEB_PKG)
	cmake --install $(BUILD_HIP) --prefix $(DEB_PKG)/usr/local
	mkdir -p $(DEB_PKG)/DEBIAN
	printf 'Package: llama-cpp-turboquant-hip-gfx1103\nVersion: $(VERSION)\nArchitecture: amd64\nMaintainer: Value-Added International <value.added.kr@gmail.com>\nDepends: amdrocm-core7.13\nDescription: llama.cpp + TurboQuant+ HIP/ROCm 7.13 gfx1103\n Port of llama.cpp with TurboQuant+ KV-cache and weight quantization,\n built for AMD Radeon 780M (gfx1103 RDNA3) with ROCm 7.13.\n' > $(DEB_PKG)/DEBIAN/control
	printf '#!/bin/sh\nldconfig\n' > $(DEB_PKG)/DEBIAN/postinst
	printf '#!/bin/sh\nldconfig\n' > $(DEB_PKG)/DEBIAN/postrm
	chmod 755 $(DEB_PKG)/DEBIAN/postinst $(DEB_PKG)/DEBIAN/postrm
	fakeroot dpkg-deb --build $(DEB_PKG)
	@printf "\nPackage ready: $(DEB_PKG).deb\n"
