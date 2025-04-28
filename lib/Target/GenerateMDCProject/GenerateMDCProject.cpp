// dfg-mlir/lib/Target/GenerateMDCProject/GenerateMDCProject.cpp
#include "dfg-mlir/Target/GenerateMDCProject/GenerateMDCProjectEmitter.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h" 
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "mlir/IR/BuiltinOps.h"          
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Dialect/Arith/IR/Arith.h" // Include the header for AddIOp
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
#include <mlir/IR/Builders.h>
#include <mlir/Support/LogicalResult.h>
#include <string>
#include "llvm/Support/FileSystem.h"
#include <fstream>

namespace mlir {
    namespace dfg {

        namespace {
            static constexpr llvm::StringRef calPackageName = "custom";
            // Helper function to get the index of an operand
            unsigned getOperandIndex(mlir::Value operand) {
                if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(operand)) {
                    return blockArg.getArgNumber();
                } else {
                    // If it's not a BlockArgument, it's likely a result from a previous operation
                    // Here we assume that the operand is a result of a previous op
                    // If needed, you can add extra checks for more complex cases
                    return operand.getDefiningOp()->getResult(0).cast<mlir::BlockArgument>().getArgNumber();
                }
            }

            LogicalResult generateCALFile(ProcessOp op, const std::string& outputDir) {
                std::string actorName = op.getSymName().str();
                if (!actorName.empty() && actorName[0] == '@') actorName = actorName.substr(1);
                llvm::SmallString<128> filepath(outputDir);
                llvm::sys::path::append(filepath, actorName + ".cal");
                llvm::errs() << "[INFO] Creating CAL file: " << filepath << "\n";

                std::error_code ec;
                llvm::raw_fd_ostream os(filepath, ec);
                //llvm::errs() << "Found process: " << op.getSymName() << "\n";
                if (ec) {
                    llvm::errs() << "[ERROR] Failed to open file " << filepath << ": " << ec.message() << "\n";
                    return op.emitError("Failed to create CAL file: ") << ec.message();
                }

                llvm::errs() << "[INFO] Writing CAL contents for: " << actorName << "\n";
                
                os << "package "<<calPackageName<<";\n\n";
                os << "actor " << op.getSymName() << "()\n";
                
                // Input ports
                auto funcType = op.getFunctionType();
                for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
                    os << "  uint(size=32) a" << i;
                    os << (i + 1 < funcType.getNumInputs() ? ",\n" : "\n");
                                }
                
                // Output ports
                os << "  ==>\n";
                for (unsigned i = 0; i < funcType.getNumResults(); ++i) {
                    os << "  uint(size=32) c" << i;
                    os << (i + 1 < funcType.getNumResults() ? ",\n" : "\n");
                }
                
                // Simple action
                os << ":\n";
                os << "  action ";
                for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
                    os << "a" << i << ":[x" << i << "]";
                    os << (i + 1 < funcType.getNumInputs() ? ", " : " ");
                }
                os << "==> ";
                unsigned outputIdx = 0;
                op.getBody().walk([&](mlir::Operation *innerOp) {
                    auto handleOperand = [&](mlir::Value val) -> unsigned {
                        if (auto blockArg = val.dyn_cast<mlir::BlockArgument>()) {
                            return blockArg.getArgNumber();
                        } else if (auto pullOp = llvm::dyn_cast<dfg::PullOp>(val.getDefiningOp())) {
                            auto originalInput = pullOp.getOperand();
                            if (auto origBlockArg = originalInput.dyn_cast<mlir::BlockArgument>()) {
                                return origBlockArg.getArgNumber();
                            }
                        }
                        llvm::errs() << "[ERROR] Unexpected operand type!\n";
                        return -1; // invalid
                    };
                
                    if (auto addOp = llvm::dyn_cast<mlir::arith::AddIOp>(innerOp)) {
                        os << "c" << outputIdx << ":[x" 
                           << handleOperand(addOp.getLhs()) 
                           << " + x" 
                           << handleOperand(addOp.getRhs())
                           << "]";
                        outputIdx++;
                    } else if (auto mulOp = llvm::dyn_cast<mlir::arith::MulIOp>(innerOp)) {
                        os << "c" << outputIdx << ":[x" 
                           << handleOperand(mulOp.getLhs()) 
                           << " * x" 
                           << handleOperand(mulOp.getRhs())
                           << "]";
                        outputIdx++;
                    } else if (auto subOp = llvm::dyn_cast<mlir::arith::SubIOp>(innerOp)) {
                        os << "c" << outputIdx << ":[x" 
                           << handleOperand(subOp.getLhs()) 
                           << " - x" 
                           << handleOperand(subOp.getRhs())
                           << "]";
                        outputIdx++;
                    }
                
                    if (outputIdx + 1 < funcType.getNumResults()) {
                        os << ", ";
                    }
                });
                
                os << "\n  end\n";
                os << "end\n";
                llvm::errs() << "[INFO] Successfully wrote CAL file: " << filepath << "\n";
             
                return success();
            }
            } // namespace

LogicalResult generateMDCProject(Operation* op, const std::string& outputDir) {
        std::string outputDirMDC = outputDir + "MDC/";
        
        if (!llvm::sys::fs::exists(outputDirMDC)) {
            std::error_code ec = llvm::sys::fs::create_directory(outputDirMDC);
            if (ec) {
                return op->emitError("Failed to create package directory: ")
                       << outputDirMDC << " (" << ec.message() << ")";
            }
        }
        if (!llvm::sys::fs::exists(outputDirMDC+"bin/")) {
            std::error_code ec = llvm::sys::fs::create_directory(outputDirMDC+"bin/");
            if (ec) {
                return op->emitError("Failed to create package directory: ")
                       << outputDirMDC+"bin/" << " (" << ec.message() << ")";
            }
        }
        if (!llvm::sys::fs::exists(outputDirMDC+"reference/")) {
            std::error_code ec = llvm::sys::fs::create_directory(outputDirMDC+"reference/");
            if (ec) {
                return op->emitError("Failed to create package directory: ")
                       << outputDirMDC +"reference/"<< " (" << ec.message() << ")";
            }
        }
        outputDirMDC = outputDirMDC + "src/";
        if (!llvm::sys::fs::exists(outputDirMDC)) {
            std::error_code ec = llvm::sys::fs::create_directory(outputDirMDC);
            if (ec) {
                return op->emitError("Failed to create package directory: ")
                       << outputDirMDC << " (" << ec.message() << ")";
            }
        }

        std::string caloutputDir = outputDirMDC + calPackageName.str();
        outputDirMDC = outputDirMDC + "baseline/";
        if (!llvm::sys::fs::exists(outputDirMDC)) {
            std::error_code ec = llvm::sys::fs::create_directory(outputDirMDC);
            if (ec) {
                return op->emitError("Failed to create package directory: ")
                       << outputDirMDC << " (" << ec.message() << ")";
            }
        }
        llvm::SmallString<128> xdfPath(outputDirMDC);
        llvm::sys::path::append(xdfPath, "top.xdf");

        if (!llvm::sys::fs::exists(caloutputDir)) {
            std::error_code ec = llvm::sys::fs::create_directory(caloutputDir);
            if (ec) {
                return op->emitError("Failed to create CAL package directory: ")
                       << caloutputDir << " (" << ec.message() << ")";
            }
        }

        std::error_code ec;
        llvm::raw_fd_ostream os(xdfPath, ec);
        if (ec) {
            llvm::errs() << "[ERROR] Failed to open file " << xdfPath << ": " << ec.message() << "\n";
            return op->emitError("Failed to open XDF output file: ")
            << ec.message();
        }
        llvm::errs() << "=== Starting MDC Project Generation ===\n";
        LogicalResult calResult = success();
        unsigned procCount = 0;
        llvm::errs() << "[INFO] Scanning for ProcessOps...\n";
        op->walk([&](ProcessOp proc) {
            llvm::errs() << "[INFO] Found ProcessOp: " << proc.getSymName() << "\n";
            ++procCount;
            if (failed(generateCALFile(proc, caloutputDir))) {
                llvm::errs() << "[ERROR] Failed to generate CAL file for: " << proc.getSymName() << "\n";
                calResult = failure();
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });
        if (procCount == 0) {
            llvm::errs() << "[WARN] No ProcessOps found in the IR.\n";
        }
        if (failed(calResult)) return failure();
        
    // Start XDF document
    os << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>";
    os << "<XDF name=\"top\">\n";
    // Process ModuleOp inputs/outputs
    if (auto module = dyn_cast<ModuleOp>(op)) {
        auto topRegion = module.lookupSymbol<RegionOp>("top");
        if (topRegion) {
            llvm::errs() << "[INFO] Found dfg.region 'top' with inputs/outputs\n";            
            auto inputTypes = topRegion.getFunctionType().getInputs();
            auto outputTypes = topRegion.getFunctionType().getResults();
        
            // Input ports
            for (unsigned i = 0; i < inputTypes.size(); ++i) {
                os << "  <Port kind=\"Input\" name=\"arg" << i << "\">\n";
                os << "    <Type name=\"int\">\n";
                os << "      <Entry kind=\"Expr\" name=\"size\">\n";
                os << "        <Expr kind=\"Literal\" literal-kind=\"Integer\" value=\"32\"/>\n";
                os << "      </Entry>\n";
                os << "    </Type>\n";
                os << "  </Port>\n";
            }
        
            // Output ports
            for (unsigned i = 0; i < outputTypes.size(); ++i) {
                os << "  <Port kind=\"Output\" name=\"arg" << (inputTypes.size() + i) << "\">\n";
                os << "    <Type name=\"int\">\n";
                os << "      <Entry kind=\"Expr\" name=\"size\">\n";
                os << "        <Expr kind=\"Literal\" literal-kind=\"Integer\" value=\"32\"/>\n";
                os << "      </Entry>\n";
                os << "    </Type>\n";
                os << "  </Port>\n";
            }
            // Create a single instance (assuming one process for now)
            unsigned instNum = 0;
            unsigned inputArgIndex = 0;
            unsigned outputArgIndex = inputTypes.size(); // Outputs start after all inputs
            topRegion.getBody().walk([&](InstantiateOp inst) {
                std::string instName = inst.getCallee().getRootReference().str() + std::to_string(instNum++);
                os << "  <Instance id=\"" << instName << "\">\n";
                os << "    <Class name=\""<<calPackageName<<"." << inst.getCallee().getRootReference().str() << "\"/>\n";
                os << "  </Instance>\n";
    
                // Connect inputs
                for (unsigned i = 0; i < inst.getInputs().size(); ++i) {
                    os << "  <Connection dst=\"" << instName << "\" dst-port=\"a" << i
                       << "\" src=\"\" src-port=\"arg" << inputArgIndex++ << "\"/>\n";
                }
    
                // Connect outputs
                for (unsigned i = 0; i < inst.getOutputs().size(); ++i) {
                    os << "  <Connection dst=\"\" dst-port=\"arg" << outputArgIndex++
                       << "\" src=\"" << instName << "\" src-port=\"c" << i << "\"/>\n";
                }
            });            
        }
        
        else {
            llvm::errs() << "[ERROR] No dfg.region @top found.\n";
        }      
    }else {
        llvm::errs() << "[ERROR] Top-level op is not a ModuleOp.\n";
    }
  

    os << "</XDF>\n";    
    // Generate top.xdfdiag
    std::string xdfDiagPath = outputDirMDC + "top.xdfdiag";
    std::ofstream diagFile(xdfDiagPath);
    if (!diagFile.is_open()) {
    llvm::errs() << "Error: Could not open " << xdfDiagPath << " for writing.\n";
    return mlir::failure();
    }

    diagFile << R"(<?xml version="1.0" encoding="ASCII"?>
    <pi:Diagram xmi:version="2.0" xmlns:xmi="http://www.omg.org/XMI" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:al="http://eclipse.org/graphiti/mm/algorithms" xmlns:pi="http://eclipse.org/graphiti/mm/pictograms" visible="true" gridUnit="10" diagramTypeId="xdfDiagram" name="top" snapToGrid="true" version="0.18.0">
    <graphicsAlgorithm xsi:type="al:Rectangle" background="//@colors.1" foreground="//@colors.0" lineWidth="1" transparency="0.0" width="1000" height="1000"/>
    <colors red="227" green="238" blue="249"/>
    <colors red="255" green="255" blue="255"/>
    </pi:Diagram>
    )";
    diagFile.close();
    llvm::outs() << "Generated: " << xdfDiagPath << "\n";
    
    return success();
}

} // namespace dfg
} // namespace mlir