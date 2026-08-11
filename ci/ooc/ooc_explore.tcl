# G8σ rescue/diagnostic run (cycle-identical levers only): retiming +
# area directives + aggressive phys_opt, and a hierarchical util dump.
# Usage: vivado -mode batch -source ooc_explore.tcl -tclargs <top> <srcs...>
set top  [lindex $argv 0]
set srcs [lrange $argv 1 end]
set repo_root [file normalize [file join [file dirname [info script]] ../..]]
set part xck26-sfvc784-2LV-c
set period 8.000
create_project -in_memory -part $part
foreach f $srcs { read_verilog -sv $f }
if {$top eq "tensor_sidecar"} {
  set tensor_lut_dir [file join $repo_root rtl tensor]
  set_property generic "LUT_DIR=\"$tensor_lut_dir/\"" [current_fileset]
}
if {$top eq "s_cluster"} {
  set_property generic {NB=10} [current_fileset]
}
if {$top eq "fabric_grid"} {
  set_property generic {NB=12} [current_fileset]
}
# D-025: the core is its own 50 MHz domain
if {$top eq "simt_core"} { set period 20.000 }
synth_design -top $top -mode out_of_context \
  -directive AreaOptimized_high -retiming
create_clock -period $period -name clk [get_ports clk]
report_utilization -hierarchical -file ./${top}_explore_hier.rpt
opt_design -directive ExploreArea
place_design -directive ExtraTimingOpt
phys_opt_design -directive AggressiveExplore
route_design -directive AggressiveExplore
# reports + checkpoint IMMEDIATELY post-route: a later crash must not
# eat a computed result (learned 2026-07-11: Vivado aborted between
# "Routing Is Done" and these lines, losing the verdict)
write_checkpoint -force ./${top}_postroute.dcp
check_timing -file ./${top}_explore_check.rpt
report_timing_summary -file ./${top}_explore_route.rpt
report_utilization -file ./${top}_explore_util.rpt
set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
puts "G8X_WNS ${top} ${wns}"
# bonus pass: post-route phys_opt (where the 07-11 crash lived)
phys_opt_design -directive AggressiveExplore
report_timing_summary -file ./${top}_explore_route2.rpt
set wns2 [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
puts "G8X_WNS2 ${top} ${wns2}"
