# Darwin-25 CLT damage: clang lost its own libc++ headers; point at the SDK copy
export CPLUS_INCLUDE_PATH := /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
MK_ROOT := ../..

DUT         := pdit_cell
TEST_MODULE := test_pdit_cell
FUZZ_N      := 1000
VERILOG_SOURCES := $(MK_ROOT)/rtl/pbit/pdit_cell.sv

# $readmemh resolves relative to the sim run dir — link the golden-generated LUT
exp_lut.mem:
	ln -sf $(MK_ROOT)/rtl/pbit/exp_lut.mem exp_lut.mem

smoke fuzz: exp_lut.mem

include ../common.mk
