# Darwin-25 CLT damage: clang lost its own libc++ headers; point at the SDK copy
export CPLUS_INCLUDE_PATH := /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
MK_ROOT := ../..

DUT         := port_arbiter
TEST_MODULE := test_port_arbiter
FUZZ_N      := 8000
VERILOG_SOURCES := $(MK_ROOT)/rtl/mem/port_arbiter.sv
# the deployed simt_soc shape (D-032c leg C)
COMPILE_ARGS += -GCRED=4 -GCRED_B=4 -GCRED_C=4

include ../common.mk
