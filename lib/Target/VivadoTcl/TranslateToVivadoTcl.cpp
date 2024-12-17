//===- TranslateToVivadoTcl.cpp - Translating to Vivado Tcl
//-----------------===//
//
//===----------------------------------------------------------------------===//

#include "dfg-mlir/Dialect//dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "dfg-mlir/Target/VivadoTcl/VivadoTclEmitter.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Support/LogicalResult.h>
#include <string>

namespace mlir {
namespace dfg {

llvm::DenseMap<Value, std::string> regionIODmaPorts;
llvm::DenseMap<Value, std::string> channelIOFifoPorts;
llvm::StringMap<int> instanceNums;

LogicalResult printOperation(ChannelOp op, raw_ostream &os)
{
    int numFifo;
    if (instanceNums.count("fifo")) {
        numFifo = instanceNums["fifo"]++;
    } else {
        numFifo = 0;
        instanceNums["fifo"] = 1;
    }
    os << "set fifo_" << numFifo
       << " [create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 "
          "fifo_"
       << numFifo << "]\n";

    auto depth = op.getBufferSize() < 16     ? 16
                 : op.getBufferSize() > 1024 ? 1024
                                             : op.getBufferSize();
    os << "set_property CONFIG.TDATA_NUM_BYTES {"
       << (int)ceil(
              op.getInChan().getType().getElementType().getIntOrFloatBitWidth()
              / 8)
       << "} $fifo_" << numFifo << "\n";
    os << "set_property CONFIG.FIFO_DEPTH {" << depth << "} $fifo_" << numFifo
       << "\n";
    os << "set_property CONFIG.HAS_TLAST {1} $fifo_" << numFifo << "\n";
    channelIOFifoPorts[op.getInChan()] =
        "fifo_" + std::to_string(numFifo) + "/S_AXIS";
    channelIOFifoPorts[op.getOutChan()] =
        "fifo_" + std::to_string(numFifo) + "/M_AXIS";
    return success();
}

LogicalResult printOperation(ConnectInputOp op, raw_ostream &os)
{
    int numConnection;
    if (instanceNums.count("connection")) {
        numConnection = instanceNums["connection"]++;
    } else {
        numConnection = 0;
        instanceNums["connection"] = 1;
    }
    os << "connect_bd_intf_net -intf_net connection_" << numConnection;
    os << " [get_bd_intf_pins " << regionIODmaPorts[op.getRegionPort()] << "] ";
    os << "[get_bd_intf_pins " << channelIOFifoPorts[op.getChannelPort()]
       << "]";
    return success();
}

LogicalResult printOperation(ConnectOutputOp op, raw_ostream &os)
{
    int numConnection;
    if (instanceNums.count("connection")) {
        numConnection = instanceNums["connection"]++;
    } else {
        numConnection = 0;
        instanceNums["connection"] = 1;
    }
    os << "connect_bd_intf_net -intf_net connection_" << numConnection;
    os << " [get_bd_intf_pins " << channelIOFifoPorts[op.getChannelPort()]
       << "] ";
    os << "[get_bd_intf_pins " << regionIODmaPorts[op.getRegionPort()] << "]";
    return success();
    return success();
}

LogicalResult printOperation(InstantiateOp op, raw_ostream &os)
{
    auto instanceName = op.getCallee().getRootReference().str();
    int numInstance;
    if (instanceNums.count(instanceName)) {
        numInstance = instanceNums[instanceName]++;
    } else {
        numInstance = 0;
        instanceNums[instanceName] = 1;
    }
    os << "set " << instanceName << "_" << numInstance
       << " [create_bd_cell -type ip -vlnv xilinx.com:hls:" << instanceName
       << ":1.0 " << instanceName << "_" << numInstance << "]\n";

    if (!instanceNums.count("connection")) instanceNums["connection"] = 0;
    int idx = 1;
    for (auto input : op.getInputs()) {
        os << "connect_bd_intf_net -intf_net connection_"
           << instanceNums["connection"]++;
        os << " [get_bd_intf_pins " << channelIOFifoPorts[input] << "] ";
        os << " [get_bd_intf_pins " << instanceName << "_" << numInstance
           << "/v" << idx++ << "]\n";
    }
    for (auto output : op.getOutputs()) {
        os << "connect_bd_intf_net -intf_net connection_"
           << instanceNums["connection"]++;
        os << " [get_bd_intf_pins " << instanceName << "_" << numInstance
           << "/v" << idx++ << "] ";
        os << "[get_bd_intf_pins " << channelIOFifoPorts[output] << "]\n";
    }
    return success();
}

LogicalResult emitOperation(Operation &op, raw_ostream &os)
{
    LogicalResult status =
        llvm::TypeSwitch<Operation*, LogicalResult>(&op)
            .Case<ChannelOp>([&](auto op) { return printOperation(op, os); })
            .Case<ConnectInputOp, ConnectOutputOp>(
                [&](auto op) { return printOperation(op, os); })
            .Case<InstantiateOp>(
                [&](auto op) { return printOperation(op, os); })
            .Default(
                [](auto op) { return op->emitError("unsupported operation"); });

    if (failed(status)) return failure();
    os << "\n";
    return success();
}

LogicalResult
translateToVivadoTcl(Operation* op, raw_ostream &os, std::string &targetDevice)
{
    raw_indented_ostream ios(os);

    // Project configurations
    RegionOp regionOp;
    op->walk([&regionOp](RegionOp op) { regionOp = op; });
    ios << "set project_name \"" << regionOp.getSymName() << "\"\n";
    ios << "set project_dir \"./vivado_project/$project_name\"\n";
    ios << "set project_path \"$project_dir/$project_name.xpr\"\n";
    ios << "set target_device \"" << targetDevice << "\"\n";
    ios << "set ip_repo_dir \"./hls_project\"\n";
    ios << "\n";

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
    ios << "\n";

    // Block design configurations
    ios << "set bd_name \"${project_name}_bd\"\n";
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
    ios << "set_property CONFIG.PSU__USE__M_AXI_GP2 {0} $zynq_mpsoc\n";
    ios << "set_property CONFIG.PSU__USE__S_AXI_GP0 {1} $zynq_mpsoc\n";
    ios << "set_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} "
           "$zynq_mpsoc\n";
    ios << "\n";

    // DMA IP based on the numbers of I/O of region
    auto regionFuncTy = regionOp.getFunctionType();
    auto numInputs = regionFuncTy.getNumInputs();
    auto numOutputs = regionFuncTy.getNumResults();
    int numDmaReadWrites = std::min(numInputs, numOutputs);
    for (auto i = 0; i < numDmaReadWrites; i++) {
        auto inTy = cast<OutputType>(regionFuncTy.getInput(i)).getElementType();
        auto outTy =
            cast<InputType>(regionFuncTy.getResult(i)).getElementType();
        ios << "set dma_" << i
            << " [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 dma_"
            << i << "]\n";
        ios << "set_property CONFIG.c_include_sg {0} $dma_" << i << "\n";
        ios << "set_property CONFIG.c_m_axis_mm2s_tdata_width {"
            << inTy.getIntOrFloatBitWidth() << "} $dma_" << i << "\n";
        ios << "set_property CONFIG.c_s_axis_s2mm_tdata_width {"
            << outTy.getIntOrFloatBitWidth() << "} $dma_" << i << "\n";
        regionIODmaPorts[regionOp.getBody().getArgument(i)] =
            "dma_" + std::to_string(i) + "/M_AXIS_MM2S";
        regionIODmaPorts[regionOp.getBody().getArgument(i + numInputs)] =
            "dma_" + std::to_string(i) + "/S_AXIS_S2MM";
    }
    for (auto i = 0; i < std::abs(int(numInputs - numOutputs)); i++) {
        auto bias = numDmaReadWrites + i;
        ios << "set dma_" << bias
            << " [create_bd_cell -type ip -vlnv "
               "xilinx.com:ip:axi_dma:7.1 dma_"
            << bias << "]\n";
        ios << "set_property CONFIG.c_include_sg {0} $dma_" << bias << "\n";
        if (numInputs > numOutputs) {
            auto inTy =
                cast<OutputType>(regionFuncTy.getInput(bias)).getElementType();
            ios << "set_property CONFIG.c_include_s2mm {0} $dma_" << bias
                << "\n";
            ios << "set_property CONFIG.c_m_axis_mm2s_tdata_width {"
                << inTy.getIntOrFloatBitWidth() << "} $dma_" << bias << "\n";
            regionIODmaPorts[regionOp.getBody().getArgument(bias)] =
                "dma_" + std::to_string(bias) + "/M_AXIS_MM2S";
        } else {
            auto outTy =
                cast<InputType>(regionFuncTy.getResult(bias)).getElementType();
            ios << "set_property CONFIG.c_include_mm2s {0} $dma_" << bias
                << "\n";
            ios << "set_property CONFIG.c_s_axis_s2mm_tdata_width {"
                << outTy.getIntOrFloatBitWidth() << "} $dma_" << bias << "\n";
            regionIODmaPorts[regionOp.getBody().getArgument(bias + numInputs)] =
                "dma_" + std::to_string(bias) + "/S_AXIS_S2MM";
        }
    }
    ios << "\n";

    for (auto &opi : regionOp.getBody().getOps())
        if (failed(emitOperation(opi, os)))
            return emitError(opi.getLoc(), "cannot emit this operation");
    // Apply automations
    // DMAs
    for (size_t i = 0; i < std::max(numInputs, numOutputs); i++) {
        ios << "apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { "
               "Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master "
               "{/zynq_mpsoc/M_AXI_HPM0_FPD} Slave {/dma_"
            << i
            << "/S_AXI_LITE} ddr_seg {Auto} intc_ip {New AXI Interconnect} "
               "master_apm {0}}  [get_bd_intf_pins dma_"
            << i << "/S_AXI_LITE]\n";
    }
    bool smcCreated = false;
    for (int i = 0; i < numDmaReadWrites; i++) {
        auto intcIP = smcCreated ? "/axi_smc" : "New AXI SmartConnect";
        auto getPin = smcCreated ? "dma_" + std::to_string(i) + "/M_AXI_MM2S"
                                 : "zynq_mpsoc/S_AXI_HPC0_FPD";
        ios << "apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config "
               "{ Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} "
               "Master {/dma_"
            << i
            << "/M_AXI_MM2S} Slave {/zynq_mpsoc/S_AXI_HPC0_FPD} ddr_seg "
               "{Auto} intc_ip {"
            << intcIP
            << "} master_apm {0}}  "
               "[get_bd_intf_pins "
            << getPin << "]\n";
        if (!smcCreated) smcCreated = true;
        ios << "apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config "
               "{ Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} "
               "Master {/dma_"
            << i
            << "/M_AXI_S2MM} Slave {/zynq_mpsoc/S_AXI_HPC0_FPD} ddr_seg "
               "{Auto} intc_ip {/axi_smc} master_apm {0}}  [get_bd_intf_pins "
               "dma_"
            << i << "/M_AXI_S2MM]\n";
    }
    for (int i = 0; i < std::abs(int(numInputs - numOutputs)); i++) {
        auto bias = i + std::min(numInputs, numOutputs);
        auto intcIP = smcCreated ? "/axi_smc" : "New AXI SmartConnect";
        auto direction = numInputs > numOutputs
                             ? "dma_" + std::to_string(bias) + "/M_AXI_MM2S"
                             : "dma_" + std::to_string(bias) + "/M_AXI_S2MM";
        auto getPin = smcCreated ? direction : "zynq_mpsoc/S_AXI_HPC0_FPD";
        ios << "apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config "
               "{ Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} "
               "Master {/"
            << getPin
            << "} Slave {/zynq_mpsoc/S_AXI_HPC0_FPD} ddr_seg "
               "{Auto} intc_ip {"
            << intcIP << "} master_apm {0}}  [get_bd_intf_pins " << getPin
            << "]\n";
    }
    ios << "\n";
    // FIFO clock/reset
    for (int i = 0; i < instanceNums["fifo"]; i++) {
        ios << "apply_bd_automation -rule xilinx.com:bd_rule:clkrst "
               "-config { "
               "Clk {Auto}}  [get_bd_pins fifo_"
            << i << "/s_axis_aclk]\n";
    }
    ios << "\n";
    // The HLS IPs
    for (auto &map : instanceNums) {
        auto name = map.first();
        if (name != "fifo" && name != "connection") {
            for (int i = 0; i < map.second; i++) {
                ios << "apply_bd_automation -rule xilinx.com:bd_rule:axi4 "
                       "-config { Clk_master {Auto} Clk_slave {Auto} "
                       "Clk_xbar "
                       "{Auto} Master {/zynq_mpsoc/M_AXI_HPM0_FPD} Slave {/"
                    << name << "_" << i
                    << "/s_axi_control} ddr_seg {Auto} intc_ip {New AXI "
                       "Interconnect} master_apm {0}}  [get_bd_intf_pins "
                    << name << "_" << i << "/s_axi_control]\n";
            }
        }
    }
    ios << "\n";

    // Regenerate the layout
    ios << "regenerate_bd_layout\n";
    // Validate the design
    ios << "puts \"INFO: Validating block design: $bd_name\"\n";
    ios << "validate_bd_design\n";
    // Save block design
    ios << "save_bd_design\n";
    // Close the block design
    ios << "puts \"INFO: Closing block design: $bd_name\"\n";
    ios << "close_bd_design [get_bd_designs $bd_name]\n";
    // Create a HDL wrapper
    ios << "puts \"INFO: Generating HDL wrapper for block design: $bd_name\"\n";
    ios << "make_wrapper -files [get_files "
           "\"$project_dir/$project_name.srcs/sources_1/bd/$bd_name/"
           "$bd_name.bd\"] -top\n";
    // Add wrapper as top
    ios << "set wrapper_path "
           "\"$project_dir/$project_name.srcs/sources_1/bd/$bd_name/hdl/"
           "${bd_name}_wrapper.v\"\n";
    ios << "add_files $wrapper_path\n";
    ios << "set_property top \"${bd_name}_wrapper\" [get_filesets sources_1]\n";
    ios << "\n";

    // TODO: sythn and impl
    // Get the maximum number of cores that can use for synth and impl
    ios << "set max_cores [get_param general.maxThreads]\n";
    ios << "puts \"INFO: Using up to $max_cores threads for synthesis and "
           "implementation.\"\n";
    // Synthesis
    ios << "puts \"INFO: Running Synthesis...\"\n";
    ios << "launch_runs synth_1 -jobs $max_cores\n";
    ios << "wait_on_run synth_1\n";
    // Implementation
    ios << "puts \"INFO: Running Implementation...\"\n";
    ios << "launch_runs impl_1 -jobs $max_cores\n";
    ios << "wait_on_run impl_1\n";
    // Get WNS to estimate the maximum freq can be utilized
    ios << "open_run impl_1\n";
    ios << "set wns [get_property SLACK [get_timing_paths]]\n";
    ios << "set max_freq [expr {floor(1000 / (10 - $wns))}]\n";
    ios << "close_design\n";
    // Open bd_design and configure the freq again
    ios << "open_bd_design "
           "\"$project_dir/$project_name.srcs/sources_1/bd/$bd_name/"
           "$bd_name.bd\"\n";
    ios << "set_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $max_freq "
           "[get_bd_cell zynq_mpsoc]\n";
    ios << "save_bd_design\n";
    ios << "close_bd_design [get_bd_designs $bd_name]\n";
    // Rerun synth and impl
    ios << "puts \"INFO: Rerunning Synthesis...\"\n";
    ios << "reset_run synth_1\n";
    ios << "launch_runs synth_1 -jobs $max_cores\n";
    ios << "wait_on_run synth_1\n";
    ios << "puts \"INFO: Running Implementation...\"\n";
    ios << "reset_run impl_1\n";
    ios << "launch_runs impl_1 -to_step write_bitstream -jobs $max_cores\n";
    ios << "wait_on_run impl_1\n";
    ios << "\n";

    // Export the hardware
    ios << "write_hw_platform -fixed -include_bit -force -file "
           "$project_dir/${project_name}_bd.xsa\n";
    // Exit Vivado
    ios << "exit\n";

    return success();
}

} // namespace dfg
} // namespace mlir
