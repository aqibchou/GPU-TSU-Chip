# tb/common.mk — the Σ.3 universal cocotb make include (cocotb 2.x, verilator).
# A module's tb/<mod>/Makefile sets: DUT (rtl name), TEST_MODULE, FUZZ_N — then
# includes this file. Provides `smoke` and `fuzz` targets for ci/run_units.sh.
# On a fuzz failure the SAME seed is rerun with waves; the FST + repro line are
# the failure artifact (R9: waves only persist for red runs).

MK_ROOT ?= ../..
include ../build_paths.mk

SIM ?= verilator
TOPLEVEL_LANG ?= verilog
COCOTB_TOPLEVEL ?= $(DUT)
TOPLEVEL ?= $(DUT)
COCOTB_TEST_MODULES ?= $(TEST_MODULE)
MODULE ?= $(TEST_MODULE)
VERILOG_SOURCES ?= $(MK_ROOT)/rtl/common/$(DUT).sv
SIM_BUILD ?= $(GPU_TSU_BUILD_ROOT)/$(TEST_MODULE)/sim_build

export PYTHONPATH := $(MK_ROOT):$(MK_ROOT)/tb:$(PYTHONPATH)

SMOKE_N ?= 300
FUZZ_N  ?= 10000

UNIT_TARGETS ?= smoke fuzz
TB_NAME := $(shell basename "$(CURDIR)")

.PHONY: smoke fuzz unit-targets

unit-targets:
	@echo "$(UNIT_TARGETS)"

smoke:
	@MK_SEED=0xC0FFEE MK_NVEC=$(SMOKE_N) COCOTB_TEST_FILTER=smoke \
	  $(MAKE) --no-print-directory sim

fuzz:
	@seed=$(if $(MK_FUZZ_SEED),$(MK_FUZZ_SEED),$$(date +%s)); \
	echo "fuzz seed: $$seed  n: $(FUZZ_N)"; \
	if ! MK_SEED=$$seed MK_NVEC=$(FUZZ_N) COCOTB_TEST_FILTER=fuzz \
	     $(MAKE) --no-print-directory sim; then \
	  echo "FUZZ RED — rebuilding with trace and rerunning seed $$seed"; \
	  $(MAKE) --no-print-directory clean >/dev/null; \
	  MK_SEED=$$seed MK_NVEC=$(FUZZ_N) COCOTB_TEST_FILTER=fuzz VERILATOR_TRACE=1 \
	    $(MAKE) --no-print-directory sim || true; \
	  echo "artifacts: $$(ls $(CURDIR)/dump.* $(SIM_BUILD)/dump.* 2>/dev/null)"; \
	  echo "repro:     MK_SEED=$$seed make -C tb/$(TB_NAME) fuzz  (add VERILATOR_TRACE=1 for waves)"; \
	  exit 1; \
	fi

# cocotb is needed only for the smoke/fuzz sim targets — the C++ harness
# targets (mc/dump) must build on cocotb-less hosts (Kaggle farm workers)
COCOTB_MK := $(shell cocotb-config --makefiles 2>/dev/null)
ifneq ($(COCOTB_MK),)
include $(COCOTB_MK)/Makefile.sim
endif
