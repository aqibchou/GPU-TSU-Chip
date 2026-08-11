#!/bin/bash
# Runs ON the VM. Wipes stale output FIRST (the stale-verdict trap,
# hit x3 historically), runs the three re-hearings, writes DONE last.
# The host pulls evidence then stops the instance (no self-poweroff:
# evidence before economy).
set -u
J="$HOME/rehear"
OUT="$HOME/rehear_out"
VIVADO=/opt/Xilinx/Vivado/2024.2/bin/vivado
rm -rf "$OUT"; mkdir -p "$OUT"; cd "$OUT"
ln -sf "$J"/*.mem "$OUT/" 2>/dev/null
SAMP="$J/s_cluster.sv $J/fabric_grid.sv $J/pbit_cell.sv $J/qsite_cell.sv $J/prng_farm.sv $J/xoshiro128pp.sv"
FAB="$J/fabric_grid.sv $J/pbit_cell.sv $J/qsite_cell.sv $J/prng_farm.sv $J/xoshiro128pp.sv"
run() { local top=$1 per=$2; shift 2
  echo "== $top @ ${per}ns"
  "$VIVADO" -mode batch -nolog -nojournal \
    -source "$J/rehear.tcl" -tclargs "$top" "$per" "$@" \
    2>&1 | grep -E "REHEAR_WNS|CRITICAL|ERROR" | tail -6
}
run s_cluster   8.000 $SAMP        # AW.3 deferred leg (ACCWALK v2 RTL)
run s_cluster   7.000 $SAMP        # the 7ns question, measured
run fabric_grid 7.000 $FAB         # leg-E q8 re-hearing
echo REHEAR_DONE > "$OUT/DONE"
echo ALL_DONE
