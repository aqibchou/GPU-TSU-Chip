# Darwin-25 CLT damage: clang lost its own libc++ headers; point at the SDK copy
export CPLUS_INCLUDE_PATH := /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
DUT         := fifo
TEST_MODULE := test_fifo
FUZZ_N      := 10000

include ../common.mk
