# Darwin-25 CLT damage: clang lost its own libc++ headers; point at the SDK copy
export CPLUS_INCLUDE_PATH := /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
MK_ROOT := ../..

DUT         := tensor_array
TEST_MODULE := test_tensor_array
FUZZ_N      := 200
VERILOG_SOURCES := $(MK_ROOT)/rtl/tensor/tensor_array.sv

include ../common.mk
