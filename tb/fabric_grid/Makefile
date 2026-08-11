# Darwin-25 CLT damage: clang lost its own libc++ headers; point at the SDK copy
export CPLUS_INCLUDE_PATH := /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
MK_ROOT := ../..
include ../build_paths.mk

DUT         := fabric_grid
TEST_MODULE := test_fabric_grid
FUZZ_N      := 240
VERILOG_SOURCES := $(MK_ROOT)/rtl/pbit/fabric_grid.sv $(MK_ROOT)/rtl/pbit/pbit_cell.sv \
                   $(MK_ROOT)/rtl/pbit/qsite_cell.sv \
                   $(MK_ROOT)/rtl/pbit/prng_farm.sv $(MK_ROOT)/rtl/pbit/xoshiro128pp.sv

sigmoid_lut.mem:
	ln -sf $(MK_ROOT)/rtl/pbit/sigmoid_lut.mem sigmoid_lut.mem

glut.mem:
	ln -sf $(MK_ROOT)/rtl/pbit/glut.mem glut.mem

smoke fuzz: sigmoid_lut.mem glut.mem

# --- anneal/trace harness for S3/S4 heavy runs ---
MC_DIR := $(GPU_TSU_BUILD_ROOT)/fabric_grid/mc
MC_REAL := $(MC_DIR)/grid_mc
MC := mc/grid_mc

mc: $(MC_REAL)

$(MC_REAL): $(VERILOG_SOURCES) grid_mc.cpp Makefile
	@mkdir -p $(MC_DIR)
	@rm -rf mc
	@ln -s $(MC_DIR) mc
	@ln -sf $(MK_ROOT)/rtl/pbit/sigmoid_lut.mem $(MC_DIR)/sigmoid_lut.mem
	@cp grid_mc.cpp $(MC_DIR)/grid_mc.cpp
	verilator --cc $(VERILOG_SOURCES) --exe $(MC_DIR)/grid_mc.cpp --build -j 4 \
	  --top-module fabric_grid -O3 -CFLAGS -O2 -Wall --Mdir $(MC_DIR) -o grid_mc \
	  -GLUT_FILE='"sigmoid_lut.mem"'

include ../common.mk
