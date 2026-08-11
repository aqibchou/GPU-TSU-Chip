# Darwin-25 CLT damage: clang lost its own libc++ headers; point at the SDK copy
export CPLUS_INCLUDE_PATH := /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
MK_ROOT := ../..

DUT         := dcache
TEST_MODULE := test_dcache
FUZZ_N      := 20000
VERILOG_SOURCES := $(MK_ROOT)/rtl/mem/dcache.sv

include ../common.mk
