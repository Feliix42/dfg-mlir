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
            static constexpr llvm::StringRef BaselineFolder = "baseline";
            // Utility function to create a directory if it does not exist
            static LogicalResult createDirectoryIfNeeded(StringRef path, Operation *op) {
                if (!llvm::sys::fs::exists(path)) {
                    std::error_code ec = llvm::sys::fs::create_directories(path);
                    if (ec) {
                        return op->emitError("Failed to create directory: ") << path << " (" << ec.message() << ")";
                    }
                }
                return success();
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
                
                // Action
                os << ":\n  action ";
                for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
                    os << "a" << i << ":[x" << i << "]";
                    os << (i + 1 < funcType.getNumInputs() ? ", " : " ");
                }
                os << "==> ";
                unsigned outputIdx = 0;
                op.getBody().walk([&](mlir::Operation *innerOp) {
                    auto handleOperand = [&](Value val) -> unsigned {
                        if (auto blockArg = val.dyn_cast<mlir::BlockArgument>()) {
                            return blockArg.getArgNumber();
                        }
                        if (auto pullOp = llvm::dyn_cast<dfg::PullOp>(val.getDefiningOp())) {
                            if (auto orig = pullOp.getOperand().dyn_cast<BlockArgument>()) {
                                return orig.getArgNumber();
                            }
                        }
                        llvm::errs() << "[ERROR] Unexpected operand type!\n";
                        return -1; // invalid
                    };
                
                    if (auto addOp = dyn_cast<arith::AddIOp>(innerOp)) {
                        os << "c" << outputIdx << ":[x" 
                           << handleOperand(addOp.getLhs()) 
                           << " + x" 
                           << handleOperand(addOp.getRhs())
                           << "]";
                        outputIdx++;
                    } else if (auto mulOp = llvm::dyn_cast<arith::MulIOp>(innerOp)) {
                        os << "c" << outputIdx << ":[x" 
                           << handleOperand(mulOp.getLhs()) 
                           << " * x" 
                           << handleOperand(mulOp.getRhs())
                           << "]";
                        outputIdx++;
                    } else if (auto subOp = llvm::dyn_cast<arith::SubIOp>(innerOp)) {
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
            // Helper to generate top-level XDF
            static LogicalResult generateTopXDF(StringRef outputDir, Operation *op) {
                llvm::SmallString<128> xdfPath(outputDir);
                llvm::sys::path::append(xdfPath, "top.xdf");

                std::error_code ec;
                llvm::raw_fd_ostream os(xdfPath, ec);
                if (ec) {
                    return op->emitError("Failed to create top.xdf file: ") << ec.message();
                }

                os << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
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

                llvm::errs() << "[INFO] Successfully wrote top-level XDF.\n";
                return success();
            }
            // Helper to generate top-level XDF
            static LogicalResult generateTopXDFdiag(StringRef outputDir, Operation *op) {
                llvm::SmallString<128> xdfDiagPath(outputDir);
                llvm::sys::path::append(xdfDiagPath, "top.xdfdiag");
                std::error_code ec;
                llvm::raw_fd_ostream os(xdfDiagPath, ec);
                if (ec) {
                    return op->emitError("Failed to create top.xdfdiag file: ") << ec.message();
                }
                os<< R"(<?xml version="1.0" encoding="ASCII"?>
                <pi:Diagram xmi:version="2.0" xmlns:xmi="http://www.omg.org/XMI" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:al="http://eclipse.org/graphiti/mm/algorithms" xmlns:pi="http://eclipse.org/graphiti/mm/pictograms" visible="true" gridUnit="10" diagramTypeId="xdfDiagram" name="top" snapToGrid="true" version="0.18.0">
                <graphicsAlgorithm xsi:type="al:Rectangle" background="//@colors.1" foreground="//@colors.0" lineWidth="1" transparency="0.0" width="1000" height="1000"/>
                <colors red="227" green="238" blue="249"/>
                <colors red="255" green="255" blue="255"/>
                </pi:Diagram>
                )";
                os.close();
                llvm::errs() << "[INFO] Successfully wrote top-level XDFDIAG.\n";
                return success();
            }
        
            } // namespace

LogicalResult generateMDCProject(Operation* moduleOp, const std::string& baseOutputDir) {
        llvm::errs() << "[INFO] Generating MDC project...\n";
        std::string outputRoot = baseOutputDir + "MDC/";
        if (failed(createDirectoryIfNeeded(outputRoot, moduleOp))) return failure();
        // Create subdirectories
        if (failed(createDirectoryIfNeeded(outputRoot + "bin/", moduleOp))) return failure();
        if (failed(createDirectoryIfNeeded(outputRoot + "reference/", moduleOp))) return failure();

        std::string srcDir = outputRoot + "src/";
        if (failed(createDirectoryIfNeeded(srcDir, moduleOp))) return failure();
        if (failed(createDirectoryIfNeeded(srcDir + BaselineFolder.str(), moduleOp))) return failure();

        std::string calDir = srcDir + calPackageName.str();
        if (failed(createDirectoryIfNeeded(calDir, moduleOp))) return failure();

        // Generate CAL files for each process
        moduleOp->walk([&](ProcessOp processOp) {
            if (failed(generateCALFile(processOp, calDir))) {
                moduleOp->emitError("Failed to generate CAL file for process: ") << processOp.getSymName();
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });

        // Generate top.xdf
        if (failed(generateTopXDF(srcDir + BaselineFolder.str(), moduleOp))) return failure();
        // Generate top.xdfdiag
        if (failed(generateTopXDFdiag(srcDir + BaselineFolder.str(), moduleOp))) return failure();
    
    return success();
}

} // namespace dfg
} // namespace mlir