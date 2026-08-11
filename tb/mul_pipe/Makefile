# Darwin-25 CLT damage: clang lost its own libc++ headers; point at the SDK copy
export CPLUS_INCLUDE_PATH := /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
DUT         := mul_pipe
TEST_MODULE := test_mul_pipe
FUZZ_N      := 10000

include ../common.mk
