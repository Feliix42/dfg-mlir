set project_name "top"
set project_dir "./vivado_project/"
set project_path "$project_dir/$project_name.xpr"
set target_device "xck26-sfvc784-2LV-c"
set ip_repo_dir "./hls_project"
if {[file exists $project_dir]} {
  puts "INFO: Project directory exists."
  exit 1
}
puts "INFO: Creating a new project..."
create_project $project_name $project_dir -part $target_device
set_property part $target_device [current_project]
set_property default_lib xil_defaultlib [current_project]
set_property target_language Verilog [current_project]
if {[file exists $ip_repo_dir]} {
  puts "INFO: Adding IP repository: $ip_repo_dir"
  set_property ip_repo_paths $ip_repo_dir [current_project]
} else {
  puts "WARNING: IP repository directory $ip_repo_dir does not exist!"
  exit 1
}
set bd_name "${project_name}_bd"
puts "INFO: Creating block design: $bd_name"
create_bd_design $bd_name
puts "INFO: Adding Zynq MPSoC to the block design"
set zynq_mpsoc [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_mpsoc]
puts "INFO: Applying board preset to Zynq MPSoC"
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $zynq_mpsoc
set_property CONFIG.PSU__FPGA_PL0_ENABLE {1} $zynq_mpsoc
set_property CONFIG.PSU__USE__M_AXI_GP0 {1} $zynq_mpsoc
set_property CONFIG.PSU__USE__M_AXI_GP1 {0} $zynq_mpsoc
set_property CONFIG.PSU__USE__M_AXI_GP2 {0} $zynq_mpsoc
set_property CONFIG.PSU__USE__S_AXI_GP0 {1} $zynq_mpsoc
set_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} $zynq_mpsoc
puts "INFO: Adding HLS IP kernel to the block design"
create_bd_cell -type ip -vlnv xilinx.com:hls:top:1.0 top
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/zynq_mpsoc/M_AXI_HPM0_FPD} Slave {/top/s_axi_control} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}}  [get_bd_intf_pins top/s_axi_control]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/top/m_axi_gmem_arg0} Slave {/zynq_mpsoc/S_AXI_HPC0_FPD} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}} [get_bd_intf_pins zynq_mpsoc/S_AXI_HPC0_FPD]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/top/m_axi_gmem_arg1} Slave {/zynq_mpsoc/S_AXI_HPC0_FPD} ddr_seg {Auto} intc_ip {/axi_smc_1} master_apm {0}} [get_bd_intf_pins top/m_axi_gmem_arg1]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/top/m_axi_gmem_arg2} Slave {/zynq_mpsoc/S_AXI_HPC0_FPD} ddr_seg {Auto} intc_ip {/axi_smc_1} master_apm {0}} [get_bd_intf_pins top/m_axi_gmem_arg2]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/top/m_axi_gmem_arg3} Slave {/zynq_mpsoc/S_AXI_HPC0_FPD} ddr_seg {Auto} intc_ip {/axi_smc_1} master_apm {0}} [get_bd_intf_pins top/m_axi_gmem_arg3]
puts "INFO: Validating block design: $bd_name"
if {[catch {validate_bd_design} result]} {
    puts "Block design validation failed: $result"
    exit 1
}
puts "INFO: Validating succeeded"
regenerate_bd_layout
save_bd_design
puts "INFO: Closing block design: $bd_name"
make_wrapper -files [get_files "$project_dir/$project_name.srcs/sources_1/bd/$bd_name/$bd_name.bd"] -top
set wrapper_path "$project_dir/$project_name.srcs/sources_1/bd/$bd_name/hdl/${bd_name}_wrapper.v"
add_files $wrapper_path
set_property top "${bd_name}_wrapper" [get_filesets sources_1]
set max_cores [get_param general.maxThreads]
puts "INFO: Using up to $max_cores threads for synthesis and implementation."
puts "INFO: Running Synthesis..."
launch_runs synth_1 -jobs $max_cores
wait_on_run synth_1
set synth_status [get_property STATUS [get_runs synth_1]]
if {![string match "*Complete!*" $synth_status]} {
    puts "ERROR: Synthesis failed with status: $synth_status"
    exit 1
}
puts "INFO: Synthesis completed successfully"
puts "INFO: Running Implementation..."
reset_run impl_1
launch_runs impl_1 -to_step write_bitstream -jobs $max_cores
wait_on_run impl_1
set impl_status [get_property STATUS [get_runs impl_1]]
if {![string match "*Complete!*" $impl_status]} {
    puts "ERROR: Implementation failed with status: $impl_status"
    exit 1
}
puts "INFO: Implementation completed successfully"
puts "INFO: Exporting hardware with bitstream"
write_hw_platform -fixed -include_bit -force -file $project_dir/${project_name}_bd.xsa
exit
