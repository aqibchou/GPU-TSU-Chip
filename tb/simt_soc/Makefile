# Darwin-25 CLT damage: clang lost its own libc++ headers; point at the SDK copy
export CPLUS_INCLUDE_PATH := /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
MK_ROOT := ../..
include ../build_paths.mk

VERILOG_SOURCES := $(MK_ROOT)/rtl/simt_soc.sv \
                   $(MK_ROOT)/rtl/simt/simt_core.sv \
                   $(MK_ROOT)/rtl/simt/simt_regfile.sv \
                   $(MK_ROOT)/rtl/simt/barrel_sched.sv \
                   $(MK_ROOT)/rtl/tensor/tensor_sidecar.sv \
                   $(MK_ROOT)/rtl/mem/port_arbiter.sv \
                   $(MK_ROOT)/rtl/fusion/s_cluster.sv \
                   $(MK_ROOT)/rtl/pbit/fabric_grid.sv \
                   $(MK_ROOT)/rtl/pbit/pbit_cell.sv \
                   $(MK_ROOT)/rtl/pbit/qsite_cell.sv \
                   $(MK_ROOT)/rtl/pbit/prng_farm.sv \
                   $(MK_ROOT)/rtl/pbit/xoshiro128pp.sv

# §1c profiles: PROFILE=0 union (default, build/ — unchanged path),
# 1=S (no sampler), 2=P, 3=T. Non-zero profiles build into build_pN/.
PROFILE ?= 0
BUILD := build$(if $(filter-out 0,$(PROFILE)),_p$(PROFILE))
BUILD_DIR := $(GPU_TSU_BUILD_ROOT)/simt_soc/$(BUILD)
BUILD_SOURCES := $(addprefix $(BUILD_DIR)/,$(notdir $(VERILOG_SOURCES)))

soc: $(VERILOG_SOURCES) soc_harness.cpp Makefile
	@mkdir -p $(BUILD_DIR)
	@rm -rf $(BUILD)
	@ln -s $(BUILD_DIR) $(BUILD)
	@cp soc_harness.cpp $(BUILD_DIR)/soc_harness.cpp
	@for source in $(VERILOG_SOURCES); do cp $$source $(BUILD_DIR)/; done
	verilator --cc $(BUILD_SOURCES) --exe $(BUILD_DIR)/soc_harness.cpp \
	  --threads 2 -Wno-UNOPTTHREADS --build -j 4 --top-module simt_soc -O3 \
	  -CFLAGS "-O2 -DMK_PROFILE=$(PROFILE)" --Mdir $(BUILD_DIR) \
	  -GPROFILE=$(PROFILE) \
	  -GLUT_DIR='"rtl/tensor/"' \
	  -GSIG_LUT='"rtl/pbit/sigmoid_lut.mem"' \
	  -GQ_LUT='"rtl/pbit/glut.mem"' -o soc_harness
	@echo "built: $(BUILD)/soc_harness -> $(BUILD_DIR)/soc_harness"

# units contract (ci/run_units.sh): smoke/fuzz for every tb dir.
# This dir is the persistent SoC harness (kernels need riscv-gcc, absent
# on the CI runner) — the meaningful smoke IS the full-SoC verilation
# + C++ build; deeper coverage runs in gates/ locally.
smoke: soc
fuzz: soc
unit-targets:
	@echo "smoke fuzz"
.PHONY: smoke fuzz unit-targets
