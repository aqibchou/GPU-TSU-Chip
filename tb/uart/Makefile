# Darwin-25 CLT damage: clang lost its own libc++ headers; point at the SDK copy
export CPLUS_INCLUDE_PATH := /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
DUT         := uart
TEST_MODULE := test_uart
FUZZ_N      := 400

include ../common.mk
