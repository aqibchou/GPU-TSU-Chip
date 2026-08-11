# Darwin-25 CLT damage: clang lost its own libc++ headers; point at the SDK copy
export CPLUS_INCLUDE_PATH := /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
MK_ROOT := ../..

DUT         := simt_core
TEST_MODULE := test_simt_core
FUZZ_N      := 300
VERILOG_SOURCES := $(MK_ROOT)/rtl/simt/simt_core.sv \
                   $(MK_ROOT)/rtl/simt/simt_regfile.sv \
                   $(MK_ROOT)/rtl/simt/barrel_sched.sv

include ../common.mk
