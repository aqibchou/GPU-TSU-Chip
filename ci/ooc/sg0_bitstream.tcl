# SG0 bitstream flow (docs/FPGA_IMPLEMENTATION.md#sg0-fpga-bridge): PS + sg0_top block
# design -> synth -> impl (the PR1-certified closure strategy) ->
# write_bitstream. Project mode (BD requires it); per-run DCPs are
# the preemption checkpoints.
#   vivado -mode batch -source sg0_bitstream.tcl -tclargs <profile> <srcdir>
set P      [lindex $argv 0]
set SRC    [lindex $argv 1]
set part xck26-sfvc784-2LV-c
set pname  proj_sg0_p$P
create_project $pname ./$pname -part $part -force
add_files [glob -nocomplain $SRC/*.sv $SRC/*.v]
add_files [glob $SRC/*.mem]
set_property file_type {Memory File} [get_files *.mem]

create_bd_design "sg0"
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e ps]
set_property -dict [list \
  CONFIG.PSU__USE__M_AXI_GP0 {1} \
  CONFIG.PSU__USE__M_AXI_GP1 {0} \
  CONFIG.PSU__USE__M_AXI_GP2 {0} \
  CONFIG.PSU__USE__S_AXI_GP2 {1} \
  CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} \
  CONFIG.PSU__SAXIGP2__DATA_WIDTH {32} \
  CONFIG.PSU__FPGA_PL0_ENABLE {1} \
  CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {125} \
] $ps

set sg0 [create_bd_cell -type module -reference sg0_top sg0]
set_property CONFIG.PROFILE $P $sg0

# clocking/reset
set rstgen [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst0]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins rst0/slowest_sync_clk]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins rst0/ext_reset_in]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins sg0/clk]
connect_bd_net [get_bd_pins rst0/peripheral_aresetn] [get_bd_pins sg0/rst_n]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins ps/maxihpm0_fpd_aclk]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins ps/saxihp0_fpd_aclk]

# control path: HPM0 (AXI4) -> smartconnect -> sg0 AXI-Lite
set sc0 [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect sc0]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $sc0
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins sc0/aclk]
connect_bd_net [get_bd_pins rst0/peripheral_aresetn] [get_bd_pins sc0/aresetn]
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] \
                    [get_bd_intf_pins sc0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc0/M00_AXI] \
                    [get_bd_intf_pins sg0/s_axil]

# memory path: sg0 AXI4 master -> smartconnect -> S_AXI_HP0
set sc1 [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect sc1]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $sc1
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins sc1/aclk]
connect_bd_net [get_bd_pins rst0/peripheral_aresetn] [get_bd_pins sc1/aresetn]
connect_bd_intf_net [get_bd_intf_pins sg0/m_axi] \
                    [get_bd_intf_pins sc1/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc1/M00_AXI] \
                    [get_bd_intf_pins ps/S_AXI_HP0_FPD]

assign_bd_address
validate_bd_design
save_bd_design
make_wrapper -files [get_files sg0.bd] -top
add_files -norecurse [glob ./$pname/$pname.gen/sources_1/bd/sg0/hdl/sg0_wrapper.v]
set_property top sg0_wrapper [current_fileset]

# the PR1-certified closure strategy
set_property strategy Performance_Explore [get_runs impl_1]
launch_runs synth_1 -jobs 8
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
open_run impl_1
report_utilization -file ./sg0_p${P}_util.rpt
report_timing_summary -file ./sg0_p${P}_timing.rpt
set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
file copy -force ./$pname/$pname.runs/impl_1/sg0_wrapper.bit ./sg0_p${P}.bit
puts "SG0_BIT $P $wns"
