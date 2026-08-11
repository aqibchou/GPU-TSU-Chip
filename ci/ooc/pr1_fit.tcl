# PR1 — full simt_soc routed fit per profile (profiles_card bar: the
# union tops out <= 80% on every K26 resource class; per-profile builds
# reclaim their absent domains).
#   vivado -mode batch -source pr1_fit.tcl -tclargs <PROFILE> <SAMP_NB> <src1> [src2 ...]
# Evidence-quality lessons baked in (the rehear pipe trap, hit 2026-07-16):
# utilization is reported at SYNTH and OPT — before placement can die —
# and a placement failure is caught so the reports still land. The
# runner keeps the FULL vivado log (no -nolog, no grep pipe).
set profile [lindex $argv 0]
set snb     [lindex $argv 1]
set srcs    [lrange $argv 2 end]
set part xck26-sfvc784-2LV-c
create_project -in_memory -part $part
foreach f $srcs { read_verilog -sv $f }
# device config: sampler NB per D-023; .mem files resolve in cwd
set_property generic "PROFILE=$profile SAMP_NB=$snb" [current_fileset]
synth_design -top simt_soc -mode out_of_context
report_utilization -file ./pr1_p${profile}_synth_util.rpt
create_clock -period 8.000 -name clk [get_ports clk]
opt_design
report_utilization -file ./pr1_p${profile}_opt_util.rpt
if {[catch {place_design} err]} {
  puts "PR1_PLACE_FAIL $profile $err"
} else {
  route_design
  report_utilization -file ./pr1_p${profile}_route_util.rpt
  report_timing_summary -file ./pr1_p${profile}_route.rpt
  set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
  puts "PR1_WNS $profile 8.000 $wns"
}
