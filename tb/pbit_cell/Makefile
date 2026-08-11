# Darwin-25 CLT damage: clang lost its own libc++ headers; point at the SDK copy
export CPLUS_INCLUDE_PATH := /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
MK_ROOT := ../..

DUT         := pbit_cell
TEST_MODULE := test_pbit_cell
FUZZ_N      := 2000
VERILOG_SOURCES := $(MK_ROOT)/rtl/pbit/pbit_cell.sv

# $readmemh resolves relative to the sim run dir — link the golden-generated LUT
sigmoid_lut.mem:
	ln -sf $(MK_ROOT)/rtl/pbit/sigmoid_lut.mem sigmoid_lut.mem

smoke fuzz: sigmoid_lut.mem

include ../common.mk
