# Darwin-25 CLT damage: clang lost its own libc++ headers; point at the SDK copy
export CPLUS_INCLUDE_PATH := /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
MK_ROOT := ../..

DUT         := barrel_core
TEST_MODULE := test_barrel_core
FUZZ_N      := 400
VERILOG_SOURCES := $(MK_ROOT)/rtl/simt/barrel_core.sv \
                   $(MK_ROOT)/rtl/simt/barrel_sched.sv \
                   $(MK_ROOT)/rtl/simt/warp_regfile.sv

include ../common.mk
