# Darwin-25 CLT damage: clang lost its own libc++ headers; point at the SDK copy
export CPLUS_INCLUDE_PATH := /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
MK_ROOT := ../..

DUT         := sg0_top
TEST_MODULE := test_sg0_top
UNIT_TARGETS := smoke
VERILOG_SOURCES := $(MK_ROOT)/rtl/sg0_top.v \
                   $(MK_ROOT)/rtl/sg0_bridge.sv \
                   $(MK_ROOT)/rtl/simt_soc.sv \
                   $(MK_ROOT)/rtl/simt/simt_core.sv \
                   $(MK_ROOT)/rtl/simt/simt_regfile.sv \
                   $(MK_ROOT)/rtl/simt/barrel_sched.sv \
                   $(MK_ROOT)/rtl/tensor/tensor_sidecar.sv \
                   $(MK_ROOT)/rtl/mem/port_arbiter.sv \
                   $(MK_ROOT)/rtl/fusion/s_cluster.sv \
                   $(MK_ROOT)/rtl/pbit/fabric_grid.sv \
                   $(MK_ROOT)/rtl/pbit/pbit_cell.sv \
                   $(MK_ROOT)/rtl/pbit/qsite_cell.sv \
                   $(MK_ROOT)/rtl/pbit/prng_farm.sv \
                   $(MK_ROOT)/rtl/pbit/xoshiro128pp.sv

sigmoid_lut.mem:
	ln -sf $(MK_ROOT)/rtl/pbit/sigmoid_lut.mem sigmoid_lut.mem

glut.mem:
	ln -sf $(MK_ROOT)/rtl/pbit/glut.mem glut.mem

luts:
	for f in lut_gelu lut_gelud lut_exp lut_rsqrt lut_sigmoid; do \
	  ln -sf $(MK_ROOT)/rtl/tensor/$$f.mem $$f.mem; done

smoke: sigmoid_lut.mem glut.mem luts

include ../common.mk
