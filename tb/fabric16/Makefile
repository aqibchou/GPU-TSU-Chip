# Darwin-25 CLT damage: clang lost its own libc++ headers; point at the SDK copy
export CPLUS_INCLUDE_PATH := /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
MK_ROOT := ../..
include ../build_paths.mk

DUT         := fabric16
TEST_MODULE := test_fabric16
FUZZ_N      := 400
VERILOG_SOURCES := $(MK_ROOT)/rtl/pbit/fabric16.sv $(MK_ROOT)/rtl/pbit/pbit_cell.sv \
                   $(MK_ROOT)/rtl/pbit/prng_farm.sv $(MK_ROOT)/rtl/pbit/xoshiro128pp.sv

sigmoid_lut.mem:
	ln -sf $(MK_ROOT)/rtl/pbit/sigmoid_lut.mem sigmoid_lut.mem

smoke fuzz: sigmoid_lut.mem

# --- histogram harness for the S2 10^7-sample collections ---
MC_DIR := $(GPU_TSU_BUILD_ROOT)/fabric16/mc
MC_REAL := $(MC_DIR)/fabric_mc
MC := mc/fabric_mc

mc: $(MC_REAL)

$(MC_REAL): $(VERILOG_SOURCES) fabric_mc.cpp Makefile
	@mkdir -p $(MC_DIR)
	@rm -rf mc
	@ln -s $(MC_DIR) mc
	@ln -sf $(MK_ROOT)/rtl/pbit/sigmoid_lut.mem $(MC_DIR)/sigmoid_lut.mem
	@cp fabric_mc.cpp $(MC_DIR)/fabric_mc.cpp
	verilator --cc $(VERILOG_SOURCES) --exe $(MC_DIR)/fabric_mc.cpp --build -j 4 \
	  --top-module fabric16 -O3 -CFLAGS -O2 -Wall --Mdir $(MC_DIR) -o fabric_mc \
	  -GLUT_FILE='"sigmoid_lut.mem"'

include ../common.mk
