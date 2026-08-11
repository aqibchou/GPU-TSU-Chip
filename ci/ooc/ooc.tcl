# G8σ OOC run (docs/FPGA_IMPLEMENTATION.md#timing-and-resource-requirements). Usage:
#   vivado -mode batch -source ooc.tcl -tclargs <top> <src1> [src2 ...]
set top  [lindex $argv 0]
set srcs [lrange $argv 1 end]
set repo_root [file normalize [file join [file dirname [info script]] ../..]]
set part xck26-sfvc784-2LV-c
set period 8.000
# SV2.3: the serving engine targets 250 MHz (stricter, its own bar)
if {$top eq "tensor_array"} { set period 4.000 }
create_project -in_memory -part $part
foreach f $srcs { read_verilog -sv $f }
# LUT ROM mems resolve relative to the tensor dir on the host
if {$top eq "tensor_sidecar"} {
  set tensor_lut_dir [file join $repo_root rtl tensor]
  set_property generic "LUT_DIR=\"$tensor_lut_dir/\"" [current_fileset]
}
# D-023: the Kria device config carries n <= 1024 sites (NB=10 — the
# frozen C* workload's exact size). NB=13 (8192) stays the architectural
# and simulation limit; at 13 the slot/shadow memories (1.57Mb each)
# exceed Vivado's variable-size limit (D-022).
if {$top eq "s_cluster"} {
  set_property generic {NB=10} [current_fileset]
}
# fabric Kria config under evaluation: NB=12 (4096 sites; C* n=1024 on-device)
if {$top eq "fabric_grid"} {
  set_property generic {NB=12} [current_fileset]
}
# D-025: the core is its own 50 MHz domain
if {$top eq "simt_core"} { set period 20.000 }
synth_design -top $top -mode out_of_context
create_clock -period $period -name clk [get_ports clk]
opt_design
place_design
route_design
check_timing -file ./${top}_check.rpt
report_timing_summary -file ./${top}_route.rpt
report_utilization -file ./${top}_util.rpt
set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
puts "G8_WNS ${top} ${wns}"
