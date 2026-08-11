#!/bin/bash
# SG0 bitstream runner — runs ON mk-ondemand (preemption-immune).
# Full logs per profile; DONE last; host pulls .bit + rpts then stops
# the instance. Profiles: union first (the SG0 bring-up personality),
# S and P after the union proves the flow.
set -u
export HOME=${HOME:-/root}
source /opt/Xilinx/Vivado/2024.2/settings64.sh
J="${GPU_TSU_JOB_DIR:-$HOME/sg0}"
OUT="${GPU_TSU_OUT_DIR:-$HOME/sg0_out}"
rm -rf "$OUT"; mkdir -p "$OUT"; cd "$OUT"
for P in ${SG0_PROFILES:-0}; do
  echo "== SG0 bitstream profile $P"
  vivado -mode batch -nojournal -source "$J/sg0_bitstream.tcl" \
    -tclargs "$P" "$J/src" > "$OUT/sg0_p${P}_full.log" 2>&1
  grep -E "^SG0_BIT|ERROR" "$OUT/sg0_p${P}_full.log" | tail -5 || true
done
echo SG0_DONE > "$OUT/DONE"
