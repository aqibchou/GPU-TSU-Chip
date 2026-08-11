#!/bin/bash
# G8σ driver (runs ON the DP-2 host). Emits ci/logs-style rpt files to
# $OUT and a summary line per module.
set -u
MK="${MK:-$(cd "$(dirname "$0")/../.." && pwd)}"
OUT="${OUT:-$HOME/g8_out}"
VIVADO=/opt/Xilinx/Vivado/2024.2/bin/vivado
mkdir -p "$OUT" && cd "$OUT"
run() {
  local top=$1; shift
  echo "== $top"
  "$VIVADO" -mode batch -nolog -nojournal \
    -source "$MK/ci/ooc/ooc.tcl" -tclargs "$top" "$@" \
    2>&1 | grep -E "G8_WNS|CRITICAL|ERROR" | tail -5
}
R="$MK/rtl"
run simt_core      "$R/simt/simt_core.sv" "$R/simt/simt_regfile.sv" "$R/simt/barrel_sched.sv"
run barrel_sched   "$R/simt/barrel_sched.sv"
run simt_regfile   "$R/simt/simt_regfile.sv"
run tensor_sidecar "$R/tensor/tensor_sidecar.sv"
run port_arbiter   "$R/mem/port_arbiter.sv"
run dcache         "$R/mem/dcache.sv"
ln -sf "$R"/pbit/*.mem "$OUT/" 2>/dev/null
run fabric_grid    "$R/pbit/fabric_grid.sv" "$R/pbit/pbit_cell.sv" "$R/pbit/prng_farm.sv" "$R/pbit/xoshiro128pp.sv"
run prng_farm      "$R/pbit/prng_farm.sv" "$R/pbit/xoshiro128pp.sv"
run s_cluster      "$R/fusion/s_cluster.sv" "$R/pbit/fabric_grid.sv" "$R/pbit/pbit_cell.sv" "$R/pbit/prng_farm.sv" "$R/pbit/xoshiro128pp.sv"
echo ALL_RUNS_DONE
