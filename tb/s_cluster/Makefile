# Darwin-25 CLT damage: clang lost its own libc++ headers; point at the SDK copy
export CPLUS_INCLUDE_PATH := /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
MK_ROOT := ../..

DUT         := s_cluster
TEST_MODULE := test_s_cluster
FUZZ_N      := 12
VERILOG_SOURCES := $(MK_ROOT)/rtl/fusion/s_cluster.sv \
                   $(MK_ROOT)/rtl/pbit/fabric_grid.sv \
                   $(MK_ROOT)/rtl/pbit/pbit_cell.sv \
                   $(MK_ROOT)/rtl/pbit/qsite_cell.sv \
                   $(MK_ROOT)/rtl/pbit/prng_farm.sv \
                   $(MK_ROOT)/rtl/pbit/xoshiro128pp.sv

sigmoid_lut.mem:
	ln -sf $(MK_ROOT)/rtl/pbit/sigmoid_lut.mem sigmoid_lut.mem

glut.mem:
	ln -sf $(MK_ROOT)/rtl/pbit/glut.mem glut.mem

smoke fuzz: sigmoid_lut.mem glut.mem

include ../common.mk
