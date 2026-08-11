# Assembly-era OOC re-hearing (D-034 AW.3 deferred leg + leg-E q8).
# Like ci/ooc/ooc.tcl but the clock period is caller-chosen:
#   vivado -mode batch -source rehear.tcl -tclargs <top> <period_ns> <src1> [src2 ...]
# NB generics follow ooc.tcl's frozen device configs (D-023).
set top    [lindex $argv 0]
set period [lindex $argv 1]
set srcs   [lrange $argv 2 end]
set part xck26-sfvc784-2LV-c
create_project -in_memory -part $part
foreach f $srcs { read_verilog -sv $f }
if {$top eq "s_cluster"}   { set_property generic {NB=10} [current_fileset] }
if {$top eq "fabric_grid"} { set_property generic {NB=12} [current_fileset] }
synth_design -top $top -mode out_of_context
create_clock -period $period -name clk [get_ports clk]
opt_design
place_design
route_design
check_timing -file ./${top}_p${period}_check.rpt
report_timing_summary -file ./${top}_p${period}_route.rpt
report_utilization -file ./${top}_p${period}_util.rpt
set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
puts "REHEAR_WNS ${top} ${period} ${wns}"
