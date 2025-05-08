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
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
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
#include <memory>
namespace mlir {
    namespace dfg {

        namespace {
            // Operation Handler Base Class
            class OperationHandler {
                public:
                    virtual ~OperationHandler() = default;
                    virtual LogicalResult handle(Operation* op, raw_ostream& os) = 0;
                };  
            // ProcessOp Handler for CAL generation
            class ProcessOpHandler : public OperationHandler {
                
                    // Helper lambda to convert MLIR types to CAL types
                std::string getCALType(Type type) {
                        if (auto outputType = mlir::dyn_cast<dfg::OutputType>(type)) {
                            type = outputType.getElementType();
                        }
                        else if (auto inputType = mlir::dyn_cast<dfg::InputType>(type)) {
                            type = inputType.getElementType();
                        }
                        // Handle standard types
                        if (auto intType = mlir::dyn_cast<IntegerType>(type)) {
                            return "int(size=" + std::to_string(intType.getWidth()) + ")";
                        }
                        if (auto floatType = mlir::dyn_cast<FloatType>(type)) {
                            return "int(size=" + std::to_string(floatType.getWidth()) + ")";//cal just support int
                        }

                        return ""; // Unsupported type
                    };                
            public:
                ProcessOpHandler(llvm::StringMap<int>& ){}
                    
                LogicalResult handle(Operation* op, raw_ostream& os) override {
                    auto processOp = cast<ProcessOp>(op);
                    std::string actorName = processOp.getSymName().str();
                    if (!actorName.empty() && actorName[0] == '@') {
                        actorName = actorName.substr(1);
                    }
                    
                    // Generate CAL actor declaration
                    os << "actor " << actorName << "()\n";
                    
                    // Handle inputs
                    auto funcType = processOp.getFunctionType();
                    for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
                        Type type = funcType.getInput(i);
                        std::string calType = getCALType(type);
                        if (calType.empty()) {
                            return processOp.emitError("Unsupported input type for CAL generation: ") << type;
                        }

                        os << "  " << calType << " a" << i;
                        os << (i + 1 < funcType.getNumInputs() ? ",\n" : "\n");
                    }
                    
                    // Handle outputs
                    os << "  ==>\n";
                    for (unsigned i = 0; i < funcType.getNumResults(); ++i) {
                        Type type = funcType.getResult(i);
                        std::string calType = getCALType(type);
                        if (calType.empty()) {
                            return processOp.emitError("Unsupported input type for CAL generation: ") << type;
                        }
                        os <<"  " << calType << " c" << i;
                        os << (i + 1 < funcType.getNumResults() ? ",\n" : "\n");
                    }
                    
                    // Handle actions
                    os << ":\n";
                
                    os << "end\n";
                    
                    return success();
                }
            };
            // Operation Handler Registry (CAL-specific)
            class OperationHandlerRegistry {
                llvm::DenseMap<TypeID, std::unique_ptr<OperationHandler>> handlers;
                
            public:
                template <typename OpType, typename HandlerType, typename... Args>
                void registerHandler(Args&&... args) {
                    handlers[TypeID::get<OpType>()] = 
                        std::make_unique<HandlerType>(std::forward<Args>(args)...);
                }
                
                LogicalResult handle(Operation* op, raw_ostream& os) {
                    auto it = handlers.find(op->getRegisteredInfo()->getTypeID());
                    if (it == handlers.end()) {
                        return op->emitError("unsupported operation for CAL generation");
                    }
                    return it->second->handle(op, os);
                }
            };
            // Create CAL-specific handler registry
            OperationHandlerRegistry createCALHandlerRegistry() {
                OperationHandlerRegistry registry;
                llvm::StringMap<int> instanceNums; // Only needed if you track instances
                
                registry.registerHandler<ProcessOp, ProcessOpHandler>(instanceNums);
                // Register other CAL-relevant operation handlers here
                
                return registry;
            }
            static constexpr llvm::StringRef calPackageName = "custom";
            static constexpr llvm::StringRef BaselineFolder = "baseline";
            int counter = 0;
            LogicalResult emitCALOperation(ProcessOp op, raw_ostream &os,
                                        OperationHandlerRegistry& registry) {
                os << "package " << calPackageName << ";\n\n";

                return registry.handle(op.getOperation(), os);
            }


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

            // Extract the name from an MLIR block argument or operation result
            static std::string getValueName(Value val, const std::string& defaultPrefix = "arg") {
                // MLIR doesn't store names directly on BlockArgument, so we need to
                // look at the IR representation or use debug information
                
                // For this implementation, we'll extract from the IR printing
                std::string str;
                llvm::raw_string_ostream sstream(str);
                val.print(sstream);
                
                // The printed value should have format "%name : type" or similar
                std::string printed = sstream.str();
                
                // Find the first occurrence of '%' which indicates the start of the name
                size_t percentPos = printed.find('%');
                if (percentPos != std::string::npos) {
                    // Find where the name ends (at space or colon)
                    size_t endPos = printed.find_first_of(" :", percentPos);
                    if (endPos != std::string::npos) {
                        // Extract just the name part (including the %)
                        std::string name = printed.substr(percentPos, endPos - percentPos);
                        // Remove the '%' prefix if present
                        if (!name.empty() && name[0] == '%') {
                            return name.substr(1);
                        }
                        return name;
                    }
                }
                
                // If unable to extract name, use default naming scheme
                if (auto blockArg = mlir::dyn_cast<BlockArgument>(val)) {
                    return defaultPrefix + std::to_string(blockArg.getArgNumber());
                }
                
                // For operation results, try to get a meaningful name
                if (auto definingOp = val.getDefiningOp()) {
                    return defaultPrefix + std::to_string(mlir::cast<OpResult>(val).getResultNumber());
                }
                
                return defaultPrefix;
            }

            // Helper to generate top-level XDF
            static LogicalResult generateTopXDF(StringRef outputDir, Operation *op) {
                llvm::SmallString<128> xdfPath(outputDir);
                //llvm::sys::path::append(xdfPath, "top.xdf");
                                    while (true) {
                        xdfPath.clear();
                        xdfPath.append(outputDir);
                        if (counter == 0) {
                            llvm::sys::path::append(xdfPath, "top.xdf");
                        } else {
                            llvm::sys::path::append(xdfPath, "top" + std::to_string(counter) + ".xdf");
                        }
                        
                        // Check if file exists

                        
                        if (!llvm::sys::fs::exists(xdfPath)) {
                            break; // Found available filename
                        }
                        counter++;
                    }
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
                        
                        // Get the actual input/output names from the region operation
                        std::vector<std::string> inputNames;
                        std::vector<std::string> outputNames;
                        
                        // Extract input argument names
                        for (auto &arg : topRegion.getBody().getArguments()) {
                            // Extract the name using our helper function
                            std::string argName = getValueName(arg, "in");
                            inputNames.push_back(argName);
                        }
                        
                        // Extract output names by walking through the instantiations
                        // and collecting the output names referenced in the region
                        outputNames.clear(); // Make sure we start with an empty list
                        
                        // First, identify the expected output names from the region's signature
                        // This ensures we preserve the original names from the MLIR definition
                        for (unsigned i = 0; i < outputTypes.size(); ++i) {
                            // We use the same approach as for inputs, looking at how they're defined in the region
                            std::string outName = "out" + std::to_string(i); // Default name
                            
                            // Try to find the actual name used in instantiations
                            bool found = false;
                            topRegion.getBody().walk([&](InstantiateOp inst) {
                                if (found) return WalkResult::advance();
                                
                                // Only consider if this instantiation has at least i outputs
                                if (i < inst.getOutputs().size()) {
                                    auto output = inst.getOutputs()[i];
                                    if (auto blockArg = mlir::dyn_cast<BlockArgument>(output)) {
                                        outName = getValueName(blockArg, outName);
                                        found = true;
                                        return WalkResult::interrupt();
                                    }
                                }
                                return WalkResult::advance();
                            });
                            
                            outputNames.push_back(outName);
                        }
                        
                        // Helper function to get type size
                        auto getTypeSize = [](Type type) -> unsigned {
                            if (auto outputType = mlir::dyn_cast<dfg::OutputType>(type)) {
                                type = outputType.getElementType();
                            }
                            else if (auto inputType = mlir::dyn_cast<dfg::InputType>(type)) {
                                type = inputType.getElementType();
                            }
                            
                            if (auto intType = mlir::dyn_cast<IntegerType>(type)) {
                                return intType.getWidth();
                            }
                            if (auto floatType = mlir::dyn_cast<FloatType>(type)) {
                                return floatType.getWidth();
                            }
                            return 32; // default size
                        };                    
                        
                        // Input ports with actual names
                        for (unsigned i = 0; i < inputTypes.size(); ++i) {
                            std::string portName = (i < inputNames.size()) ? inputNames[i] : "in" + std::to_string(i);
                            
                            unsigned size = getTypeSize(inputTypes[i]);
                            os << "  <Port kind=\"Input\" name=\"" << portName << "\">\n";
                            os << "    <Type name=\"" << (mlir::isa<FloatType>(inputTypes[i]) ? "int" : "int") << "\">\n"; //cal just support int
                            os << "      <Entry kind=\"Expr\" name=\"size\">\n";
                            os << "        <Expr kind=\"Literal\" literal-kind=\"Integer\" value=\"" << size << "\"/>\n";
                            os << "      </Entry>\n";
                            os << "    </Type>\n";
                            os << "  </Port>\n";
                        }
                    
                        // Output ports with actual names
                        for (unsigned i = 0; i < outputTypes.size(); ++i) {
                            std::string portName = (i < outputNames.size()) ? outputNames[i] : "out" + std::to_string(i);
                            //llvm::errs() << "[info] portname:"<<portName.substr(0,portName.length()-1)<<","<<outputNames[i]<<","<<i<<".\n";
                            unsigned size = getTypeSize(outputTypes[i]);
                            os << "  <Port kind=\"Output\" name=\"" << portName.substr(0,portName.length()-1) << "\">\n";
                            os << "    <Type name=\"" << (mlir::isa<FloatType>(outputTypes[i]) ? "int" : "int") << "\">\n"; //cal just support int
                            os << "      <Entry kind=\"Expr\" name=\"size\">\n";
                            os << "        <Expr kind=\"Literal\" literal-kind=\"Integer\" value=\"" << size << "\"/>\n";
                            os << "      </Entry>\n";
                            os << "    </Type>\n";
                            os << "  </Port>\n";
                        }
                        
                        // Create instances with connections using actual names
                        unsigned instNum = 0;
                        topRegion.getBody().walk([&](InstantiateOp inst) {
                            std::string instName = inst.getCallee().getRootReference().str() + std::to_string(instNum++);
                            os << "  <Instance id=\"" << instName << "\">\n";
                            os << "    <Class name=\""<<calPackageName<<"." << inst.getCallee().getRootReference().str() << "\"/>\n";
                            os << "  </Instance>\n";

                            // Connect inputs using the actual input names
                            for (unsigned i = 0; i < inst.getInputs().size(); ++i) {
                                Value input = inst.getInputs()[i];
                                std::string srcPortName;
                                
                                if (auto blockArg = mlir::dyn_cast<BlockArgument>(input)) {
                                    srcPortName = getValueName(blockArg, "in");
                                    
                                    os << "  <Connection dst=\"" << instName << "\" dst-port=\"a" << i
                                       << "\" src=\"\" src-port=\"" << srcPortName << "\"/>\n";
                                }
                            }

                            // Connect outputs using the actual output names
                            for (unsigned i = 0; i < inst.getOutputs().size(); ++i) {
                                Value output = inst.getOutputs()[i];
                                std::string dstPortName;
                                
                                if (auto blockArg = mlir::dyn_cast<BlockArgument>(output)) {
                                    dstPortName = getValueName(blockArg, "out");
                                    
                                    os << "  <Connection dst=\"\" dst-port=\"" << dstPortName
                                       << "\" src=\"" << instName << "\" src-port=\"c" << i << "\"/>\n";
                                }
                            }
                        });            
                    }
                    
                    else {
                        llvm::errs() << "[ERROR] No dfg.region @top found.\n";
                        return failure();
                    }      
                }else {
                    llvm::errs() << "[ERROR] Top-level op is not a ModuleOp.\n";
                    return failure();
                }
                os << "</XDF>\n";  

                llvm::errs() << "[INFO] Successfully wrote top-level XDF.\n";
                return success();
            }
            // Helper to generate top-level XDF
            static LogicalResult generateTopXDFdiag(StringRef outputDir, Operation *op) {
                llvm::SmallString<128> xdfDiagPath(outputDir);

                //llvm::sys::path::append(xdfDiagPath, "top.xdfdiag");
                if (counter == 0){
                        llvm::sys::path::append(xdfDiagPath, "top.xdfdiag");
                    } else {
                        llvm::sys::path::append(xdfDiagPath, "top" + std::to_string(counter) + ".xdfdiag");  
                    }
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

        // Create handler registry once
        auto registry = createCALHandlerRegistry();

        // Generate CAL files for each process
        moduleOp->walk([&](ProcessOp processOp) {
            // Create file for this process
            std::string actorName = processOp.getSymName().str();
            if (!actorName.empty() && actorName[0] == '@') {
                actorName = actorName.substr(1);
            }
            llvm::SmallString<128> filepath(calDir);
            llvm::sys::path::append(filepath, actorName + ".cal");
            std::error_code ec;
            llvm::raw_fd_ostream os(filepath, ec);
            if (ec) {
                processOp.emitError("Failed to create CAL file: ") << ec.message();
                return WalkResult::interrupt();
            }

            if (failed(emitCALOperation(processOp, os, registry))) {
                processOp.emitError("Failed to generate CAL content");
                return WalkResult::interrupt();
            }
            llvm::errs() << "[INFO] Successfully wrote CAL file: " << filepath << "\n";
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