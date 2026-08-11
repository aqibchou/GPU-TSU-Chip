# Darwin-25 CLT damage: clang lost its own libc++ headers; point at the SDK copy
export CPLUS_INCLUDE_PATH := /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
MK_ROOT := ../..
include ../build_paths.mk

DUT         := prng_farm
TEST_MODULE := test_prng
FUZZ_N      := 4096
VERILOG_SOURCES := $(MK_ROOT)/rtl/pbit/prng_farm.sv $(MK_ROOT)/rtl/pbit/xoshiro128pp.sv

# --- the C++ dump harness (feeds PractRand/TestU01 at native speed) ---
DUMP_DIR := $(GPU_TSU_BUILD_ROOT)/pbit_prng/dump
DUMP_REAL := $(DUMP_DIR)/prng_dump
DUMP := dump/prng_dump

dump: $(DUMP_REAL)

$(DUMP_REAL): $(MK_ROOT)/rtl/pbit/prng_farm.sv $(MK_ROOT)/rtl/pbit/xoshiro128pp.sv dump_main.cpp
	@mkdir -p $(DUMP_DIR)
	@rm -rf dump
	@ln -s $(DUMP_DIR) dump
	@cp dump_main.cpp $(DUMP_DIR)/dump_main.cpp
	verilator --cc $(MK_ROOT)/rtl/pbit/prng_farm.sv $(MK_ROOT)/rtl/pbit/xoshiro128pp.sv \
	  --exe $(DUMP_DIR)/dump_main.cpp --build -j 4 --top-module prng_farm \
	  -O3 -CFLAGS -O2 -Wall --Mdir $(DUMP_DIR) -o prng_dump

include ../common.mk
