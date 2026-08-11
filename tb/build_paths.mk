# Verilator refuses build directories whose absolute path contains spaces.
# Keep generated C++ and objects under /tmp while sources may live anywhere.
GPU_TSU_BUILD_TAG := $(shell printf '%s' '$(CURDIR)' | cksum | awk '{print $$1}')
GPU_TSU_BUILD_ROOT ?= /tmp/gpu-tsu-chip-$(GPU_TSU_BUILD_TAG)
