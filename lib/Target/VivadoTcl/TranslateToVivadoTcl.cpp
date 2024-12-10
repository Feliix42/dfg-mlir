//===- TranslateToVivadoTcl.cpp - Translating to Vivado Tcl
//-----------------===//
//
//===----------------------------------------------------------------------===//

#include "dfg-mlir/Dialect//dfg/IR/Ops.h"
#include "dfg-mlir/Target/VivadoTcl/VivadoTclEmitter.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"

#include "llvm/ADT/StringSet.h"

#include <string>

namespace mlir {
namespace dfg {

LogicalResult
translateToVivadoTcl(Operation* op, raw_ostream &os, std::string &targetDevice)
{
    raw_indented_ostream ios(os);

    // Project configurations
    ios << "set project_dir \"./vivado_project\"\n";

    RegionOp regionOp;
    op->walk([&regionOp](RegionOp op) { regionOp = op; });
    ios << "set project_name \"" << regionOp.getSymName() << "\"\n";
    ios << "set project_path \"$project_dir/$project_name.xpr\"\n";
    ios << "set target_device \"" << targetDevice << "\"\n";
    ios << "set ip_repo_dir \"./hls_project\"\n";

    ios << "if {[file exists $project_dir]} {\n";
    ios.indent() << "puts \"INFO: Project directory exists. Deleting the "
                    "entire directory...\"\n";
    ios << "file delete -force $project_dir\n";
    ios << "puts \"INFO: Project directory deleted.\"\n";
    ios.unindent() << "}\n";
    ios << "puts \"INFO: Creating a new project...\"\n";
    ios << "create_project $project_name $project_dir -part $target_device\n";
    ios << "set_property part $target_device [current_project]\n";
    ios << "set_property default_lib xil_defaultlib [current_project]\n";
    ios << "set_property target_language Verilog [current_project]\n";

    ios << "if {[file exists $ip_repo_dir]} {\n";
    ios.indent() << "puts \"INFO: Adding IP repository: $ip_repo_dir\"\n";
    ios << "set_property ip_repo_paths $ip_repo_dir [current_project]\n";
    ios.unindent() << "} else {\n";
    ios.indent() << "puts \"WARNING: IP repository directory $ip_repo_dir does "
                    "not exist!\"\n";
    ios << "exit\n";
    ios.unindent() << "}\n";

    // Block design configurations
    ios << "set bd_name \"$project_name_top\"\n";
    ios << "puts \"INFO: Creating block design: $bd_name\"\n";
    ios << "create_bd_design $bd_name\n";

    // ZYNQ PS IP
    ios << "puts \"INFO: Adding Zynq MPSoC to the block design\"\n";
    ios << "set zynq_mpsoc [create_bd_cell -type ip -vlnv "
           "xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_mpsoc]\n";
    ios << "puts \"INFO: Applying board preset to Zynq MPSoC\"\n";
    ios << "apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e "
           "-config {apply_board_preset \"1\"} $zynq_mpsoc\n";
    ios << "set_property CONFIG.PSU__FPGA_PL0_ENABLE {1} $zynq_mpsoc\n";
    ios << "set_property CONFIG.PSU__USE__M_AXI_GP0 {1} $zynq_mpsoc\n";
    ios << "set_property CONFIG.PSU__USE__S_AXI_GP0 {1} $zynq_mpsoc\n";
    ios << "set_property CONFIG.PSU__USE__M_AXI_GP2 {0} $zynq_mpsoc\n";

    // DMA IP based on the numbers of I/O of region
    auto numInputs = regionOp.getFunctionType().getNumInputs();
    auto numOutputs = regionOp.getFunctionType().getNumResults();

    // TODO: the rest

    return success();
}

} // namespace dfg
} // namespace mlir
