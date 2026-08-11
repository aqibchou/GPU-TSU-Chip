#!/bin/bash
# PR1 runner — runs ON the VM. Full simt_soc routed fit for the union
# and each personality. FULL vivado logs kept per leg (the rehear grep
# pipe destroyed a day's evidence; never again). Wipes stale output
# FIRST (the stale-verdict trap). DONE written LAST; the host pulls
# evidence then stops the instance (evidence before economy).
set -u
J="${GPU_TSU_JOB_DIR:-$HOME/pr1}"
OUT="${GPU_TSU_OUT_DIR:-$HOME/pr1_out}"
VIVADO=/opt/Xilinx/Vivado/2024.2/bin/vivado
export HOME=${HOME:-/root}
rm -rf "$OUT"; mkdir -p "$OUT"; cd "$OUT"
ln -sf "$J"/*.mem "$OUT/" 2>/dev/null
SRCS="$J/simt_soc.sv $J/simt_core.sv $J/simt_regfile.sv \
$J/barrel_sched.sv $J/tensor_sidecar.sv $J/port_arbiter.sv \
$J/s_cluster.sv $J/fabric_grid.sv $J/pbit_cell.sv $J/qsite_cell.sv \
$J/prng_farm.sv $J/xoshiro128pp.sv"
for P in 0 1 2; do
  echo "== PR1 profile $P (SAMP_NB=10)"
  "$VIVADO" -mode batch -nojournal \
    -source "$J/pr1_fit.tcl" -tclargs "$P" 10 $SRCS \
    > "$OUT/pr1_p${P}_full.log" 2>&1
  grep -E "PR1_WNS|PR1_PLACE_FAIL" "$OUT/pr1_p${P}_full.log" || true
done
echo PR1_DONE > "$OUT/DONE"
echo ALL_DONE
