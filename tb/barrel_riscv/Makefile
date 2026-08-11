# Darwin-25 CLT damage: clang lost its own libc++ headers; point at the SDK copy
export CPLUS_INCLUDE_PATH := /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
MK_ROOT := ../..

DUT         := barrel_core
TEST_MODULE := test_barrel_riscv
FUZZ_N      := 1
# This suite requires an MK_ELF image and is driven by gates/simt_dod.py.
# An unfiltered unit invocation would be meaningless without that input.
UNIT_TARGETS :=
VERILOG_SOURCES := $(MK_ROOT)/rtl/simt/barrel_core.sv \
                   $(MK_ROOT)/rtl/simt/barrel_sched.sv \
                   $(MK_ROOT)/rtl/simt/warp_regfile.sv
COMPILE_ARGS += -GRESET_PC=32\'h80000000

include ../common.mk
