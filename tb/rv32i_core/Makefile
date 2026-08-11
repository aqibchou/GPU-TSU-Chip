# Darwin-25 CLT damage: clang lost its own libc++ headers; point at the SDK copy
export CPLUS_INCLUDE_PATH := /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
# rv32i_core lockstep bench — program-driven trace diff, NOT the cocotb Σ.3
# pattern (see docs/SOFTWARE_AND_VALIDATION.md#testbench-methodology: this is the trace-diff sub-pattern).
# smoke = fixed-seed 500-instr program through RTL+ISS+spike, 3-way diff.
# fuzz  = one 20k-instr chunk, seed from MK_FUZZ_SEED (nightly passes epoch).
# The full P1 bar (all rv32ui + 1M instrs) runs in gates/p1_lockstep.py --nightly.

MK_ROOT := ../..
include ../build_paths.mk
PY      ?= $(if $(wildcard $(MK_ROOT)/.venv/bin/python),$(MK_ROOT)/.venv/bin/python,python3)
GATE    := $(MK_ROOT)/gates/p1_lockstep.py
VSRC    := $(MK_ROOT)/rtl/simt/rv32i_core.sv
OBJ_DIR := $(GPU_TSU_BUILD_ROOT)/rv32i_core/obj_dir
SIM_REAL := $(OBJ_DIR)/Vrv32i_core
SIM     := obj_dir/Vrv32i_core

.PHONY: build smoke fuzz unit-targets clean

build: $(SIM_REAL)

$(SIM_REAL): $(VSRC) harness.cpp
	@mkdir -p $(OBJ_DIR)
	@rm -rf obj_dir
	@ln -s $(OBJ_DIR) obj_dir
	@cp harness.cpp $(OBJ_DIR)/harness.cpp
	verilator --cc $(VSRC) --exe $(OBJ_DIR)/harness.cpp --build -j 4 \
	  --top-module rv32i_core -O3 -CFLAGS -O2 -Wall --Mdir $(OBJ_DIR)

smoke: $(SIM_REAL)
	@$(PY) $(GATE) --fuzz 1 --size 500 --seed 0xC0FFEE

fuzz: $(SIM_REAL)
	@$(PY) $(GATE) --fuzz 1 --size 20000 $(if $(MK_FUZZ_SEED),--seed $(MK_FUZZ_SEED),)

unit-targets:
	@echo "smoke fuzz"

clean:
	rm -rf obj_dir $(OBJ_DIR)
