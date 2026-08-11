# Darwin-25 CLT damage: clang lost its own libc++ headers; point at the SDK copy
export CPLUS_INCLUDE_PATH := /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
MK_ROOT := ../..

DUT         := sg0_bridge
TEST_MODULE := test_sg0_bridge
UNIT_TARGETS := smoke
VERILOG_SOURCES := $(MK_ROOT)/rtl/sg0_bridge.sv

smoke:

include ../common.mk
