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
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
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
                llvm::StringMap<int>& instanceNums;
                unsigned intermediateVarCount = 0;
                    // Helper lambda to convert MLIR types to CAL types
                std::string getCALType(Type type) {
                        if (auto outputType = type.dyn_cast<dfg::OutputType>()) {
                            type = outputType.getElementType();
                        }
                        else if (auto inputType = type.dyn_cast<dfg::InputType>()) {
                            type = inputType.getElementType();
                        }
                        // Handle standard types
                        if (auto intType = type.dyn_cast<IntegerType>()) {
                            return "int(size=" + std::to_string(intType.getWidth()) + ")";
                        }
                        if (auto floatType = type.dyn_cast<FloatType>()) {
                            return "float(size=" + std::to_string(floatType.getWidth()) + ")";
                        }

                        return ""; // Unsupported type
                    };                
            public:
                ProcessOpHandler(llvm::StringMap<int>& nums) 
                    : instanceNums(nums) {}
                    
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
                    handleActions(processOp, os, funcType);
                    
                    os << "end\n";
                    
                    return success();
                }
                
            private:
                struct IntermediateVar {
                    std::string name;
                    std::string type;
                    Value definingValue;
                };
                void handleActions(ProcessOp op, raw_ostream& os, FunctionType funcType) {
                    llvm::DenseMap<Value, std::string> valueToVarName;
                    llvm::SmallVector<IntermediateVar> intermediateVars;
                    // Forward declare the function object
                    std::function<std::string(Value)> getValueExpr;
                    // First generate input patterns
                    /*for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
                        os << "a" << i << ":[x" << i << "]";
                        if (i + 1 < funcType.getNumInputs()) {
                            os << ", ";
                        }
                    }
                    
                    os << " ==> ";
                    
                    // Then generate output expressions
                    unsigned outputIdx = 0;*/
                    op.getBody().walk([&](mlir::Operation *innerOp) {
                                if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp, 
                                    arith::DivUIOp, arith::RemSIOp, arith::RemUIOp, arith::AddFOp,
                                    arith::SubFOp, arith::MulFOp, arith::DivFOp, arith::RemFOp,
                                    index::AddOp, index::SubOp, index::MulOp, index::RemSOp,
                                    index::RemUOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp,
                                    arith::ShLIOp, arith::ShRSIOp, arith::CmpIOp, arith::CmpFOp>(innerOp)) {
                                // Only consider operations that produce results
                                if (innerOp->getNumResults() > 0) {
                                    Value result = innerOp->getResult(0);
                                    // Skip if this is directly pushed to an output
                                    bool isOutput = false;
                                    for (auto &use : result.getUses()) {
                                        if (isa<dfg::PushOp>(use.getOwner())) {
                                            isOutput = true;
                                            break;
                                        }
                                    }
                                    if (!isOutput) {
                                        std::string varName = "z" + std::to_string(intermediateVars.size());
                                        valueToVarName[result] = varName;
                                        intermediateVars.push_back({
                                            varName,
                                            getCALType(result.getType()),
                                            result
                                        });
                                    }
                                }
                            }
                        });
                        // Declare intermediate variables
                        for (const auto &var : intermediateVars) {
                            os << "  " << var.type << " " << var.name << ";\n";
                        }
                        // Generate action
                        os << "  action ";
                        // Input patterns
                        for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
                            os << "a" << i << ":[x" << i << "]";
                            if (i + 1 < funcType.getNumInputs()) {
                                os << ", ";
                            }
                        }
                        os << " ==> ";
                        //unsigned outputIdx = 0;
                        llvm::DenseMap<Value, unsigned> outputMap; // Maps output block args to indices
                        for (auto arg : op.getBody().getArguments()) {
                            if (arg.getType().isa<dfg::OutputType>()) {
                                outputMap[arg] = outputMap.size();
                            }
                        }
                        llvm::errs() << "[DEBUG] Starting output expression generation\n";
                        llvm::SmallVector<bool> outputsProcessed(funcType.getNumResults(), false);
                        bool firstOutput = true;

                        op.getBody().walk([&](mlir::Operation *innerOp) {
                            if (auto pushOp = dyn_cast<dfg::PushOp>(innerOp)) {
                                llvm::errs() << "[DEBUG] Found PushOp: " << *pushOp << "\n";
                                // operand(0) is the value being pushed
                                Value pushedValue = pushOp->getOperand(0);
                                // operand(1) is the output port
                                Value outputArg = pushOp->getOperand(1);

                                unsigned outputIdx = outputArg.cast<BlockArgument>().getArgNumber() - funcType.getNumInputs();
                                if (!firstOutput) os << ", ";
                                firstOutput = false;

                                llvm::errs() << "[DEBUG] Output index: " << outputIdx << ", Pushed value: " << pushedValue << "\n";
                                if (outputIdx >= funcType.getNumResults()) {
                                    llvm::errs() << "[ERROR] Invalid output index: " << outputIdx << "\n";
                                    return;
                                }
                                if (outputsProcessed[outputIdx]) {
                                    llvm::errs() << "[WARNING] Output index already processed: " << outputIdx << "\n";
                                    return;
                                }


                                //if (valueToVarName.count(pushedValue)) {
                                    //llvm::errs() << "[DEBUG] Using intermediate var: " << valueToVarName[pushedValue] << "\n";
                                    //os << "c" << outputIdx << ":[" << valueToVarName[pushedValue] << "]";
                                //} else {
                                    // Handle direct input to output case
                                    //llvm::errs() << "[DEBUG] Generating direct expression\n";
                                    getValueExpr = [&](Value val) -> std::string {
                                        if (auto blockArg = val.dyn_cast<BlockArgument>()) {
                                            return "x" + std::to_string(blockArg.getArgNumber());
                                        }
                                        if (auto pullOp = dyn_cast<dfg::PullOp>(val.getDefiningOp())) {
                                            Value pullOperand = pullOp.getOperand();
                                            if (auto orig = pullOperand.dyn_cast<BlockArgument>()) {
                                                return "x" + std::to_string(orig.getArgNumber());
                                            }
                                        }
                                        if (valueToVarName.count(val)) {
                                            return valueToVarName[val];
                                        }
                                        if (auto addOp = dyn_cast<arith::AddIOp>(val.getDefiningOp())) {
                                            std::string lhs = getValueExpr(addOp.getLhs());
                                            std::string rhs = getValueExpr(addOp.getRhs());
                                            if (!lhs.empty() && !rhs.empty()) {
                                                return lhs + " + " + rhs;
                                            }
                                        }
                                        return ""; // invalid
                                    };
                                    
                                    std::string expr = getValueExpr(pushedValue);
                                    if (!expr.empty()) {
                                        os << "c" << outputIdx << ":[" << expr << "]";
                                    }
                                }
                                //outputsProcessed[outputIdx] = true;
                                //outputIdx++;
                            
                        });
                        // Generate do block if we have intermediate variables
                        if (!intermediateVars.empty()) {
                            os << "\n  do\n";
                            llvm::errs() << "[DEBUG] Generating do-block with " << intermediateVars.size() << " vars\n";
                            for (const auto &var : intermediateVars) {
                                Operation *defOp = var.definingValue.getDefiningOp();
                                auto emitAssignment = [&](const char* opSymbol, Value lhs, Value rhs) {
                                    auto handleOperand = [&](Value val) -> std::string {
                                    if (auto blockArg = dyn_cast<BlockArgument>(val)) {
                                        return "x" + std::to_string(blockArg.getArgNumber());
                                    }
                                    if (auto pullOp = dyn_cast<dfg::PullOp>(val.getDefiningOp())) {
                                        if (auto orig = dyn_cast<BlockArgument>(pullOp.getOperand())) {
                                            return "x" + std::to_string(orig.getArgNumber());
                                        }
                                    }
                                    if (valueToVarName.count(val)) {
                                        return valueToVarName[val];
                                    }
                                    return ""; // invalid
                                };
                                std::string lhsStr = handleOperand(lhs);
                                std::string rhsStr = handleOperand(rhs);
                                if (!lhsStr.empty() && !rhsStr.empty()) {
                                    os << "    " << var.name << " := " << lhsStr << " " << opSymbol << " " << rhsStr << ";\n";
                                    }
                                };
                                TypeSwitch<Operation*>(defOp)
                                    // Floating-point arithmetic
                                    .Case<arith::AddFOp>([&](auto op) { emitAssignment("+", op.getLhs(), op.getRhs()); })
                                    .Case<arith::SubFOp>([&](auto op) { emitAssignment("-", op.getLhs(), op.getRhs()); })
                                    .Case<arith::MulFOp>([&](auto op) { emitAssignment("*", op.getLhs(), op.getRhs()); })
                                    .Case<arith::DivFOp>([&](auto op) { emitAssignment("/", op.getLhs(), op.getRhs()); })
                                    .Case<arith::RemFOp>([&](auto op) { emitAssignment("mod", op.getLhs(), op.getRhs()); })

                                    // Integer arithmetic
                                    .Case<arith::AddIOp>([&](auto op) { emitAssignment("+", op.getLhs(), op.getRhs()); })
                                    .Case<arith::SubIOp>([&](auto op) { emitAssignment("-", op.getLhs(), op.getRhs()); })
                                    .Case<arith::MulIOp>([&](auto op) { emitAssignment("*", op.getLhs(), op.getRhs()); })
                                    .Case<arith::DivSIOp>([&](auto op) { emitAssignment("/", op.getLhs(), op.getRhs()); })
                                    .Case<arith::DivUIOp>([&](auto op) { emitAssignment("/", op.getLhs(), op.getRhs()); })
                                    .Case<arith::RemSIOp>([&](auto op) { emitAssignment("mod", op.getLhs(), op.getRhs()); })
                                    .Case<arith::RemUIOp>([&](auto op) { emitAssignment("mod", op.getLhs(), op.getRhs()); })
                                    .Case<index::AddOp>([&](auto op) { emitAssignment("+", op.getLhs(), op.getRhs()); })
                                    .Case<index::SubOp>([&](auto op) { emitAssignment("-", op.getLhs(), op.getRhs()); })
                                    .Case<index::MulOp>([&](auto op) { emitAssignment("*", op.getLhs(), op.getRhs()); })
                                    .Case<index::RemUOp>([&](auto op) { emitAssignment("mod", op.getLhs(), op.getRhs()); })
                                    .Case<index::RemSOp>([&](auto op) { emitAssignment("mod", op.getLhs(), op.getRhs()); })

                                    // Bitwise operations
                                    .Case<arith::AndIOp>([&](auto op) { emitAssignment("&", op.getLhs(), op.getRhs()); })
                                    .Case<arith::OrIOp>([&](auto op) { emitAssignment("|", op.getLhs(), op.getRhs()); })
                                    .Case<arith::XOrIOp>([&](auto op) { emitAssignment("^", op.getLhs(), op.getRhs()); })

                                    // Shifts need special symbols
                                    .Case<arith::ShLIOp>([&](auto op) { emitAssignment("<<", op.getLhs(), op.getRhs()); })
                                    .Case<arith::ShRSIOp>([&](auto op) { emitAssignment(">>", op.getLhs(), op.getRhs()); })
                                    .Case<arith::CmpIOp>([&](auto op) {
                                        const char* cmpOp = "";
                                        switch(op.getPredicate()) {
                                            case arith::CmpIPredicate::eq:  cmpOp = "=="; break;
                                            case arith::CmpIPredicate::ne:  cmpOp = "!="; break;
                                            case arith::CmpIPredicate::slt:
                                            case arith::CmpIPredicate::ult: cmpOp = "<"; break;
                                            case arith::CmpIPredicate::ule:
                                            case arith::CmpIPredicate::sle: cmpOp = "<="; break;
                                            case arith::CmpIPredicate::ugt:
                                            case arith::CmpIPredicate::sgt: cmpOp = ">"; break;
                                            case arith::CmpIPredicate::uge:
                                            case arith::CmpIPredicate::sge: cmpOp = ">="; break;
                                            default: cmpOp = "<=>"; // fallback
                                        }
                                        emitAssignment(cmpOp, op.getLhs(), op.getRhs()); } ) 
                                    .Case<arith::CmpFOp>([&](auto op) {
                                        const char* cmpOp = "";
                                        switch(op.getPredicate()) {
                                            case arith::CmpFPredicate::OEQ:
                                            case arith::CmpFPredicate::UEQ: cmpOp = "=="; break;
                                            case arith::CmpFPredicate::ONE:
                                            case arith::CmpFPredicate::UNE: cmpOp = "!="; break;
                                            case arith::CmpFPredicate::OGT:
                                            case arith::CmpFPredicate::UGT: cmpOp = ">"; break;
                                            case arith::CmpFPredicate::OGE:
                                            case arith::CmpFPredicate::UGE: cmpOp = ">="; break;
                                            case arith::CmpFPredicate::OLT:
                                            case arith::CmpFPredicate::ULT: cmpOp = "<"; break;
                                            case arith::CmpFPredicate::OLE:
                                            case arith::CmpFPredicate::ULE: cmpOp = "<="; break;
                                            default: cmpOp = "<=>"; // fallback
                                        }
                                        emitAssignment(cmpOp, op.getLhs(), op.getRhs()); } )                                             
                                    // ... other cases ...
                                    .Default([&](Operation*) {});   
                                    llvm::errs() << "[DEBUG] Generating assignment for " << var.name << "\n";                     
                                }
                                os << "  end\n";
                            }
                            //os << "  end\n";
                            llvm::errs() << "[DEBUG] Finished action generation\n";
                } // end of handleActions
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
            /*
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
            }*/
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
                        // Helper function to get type size
                        auto getTypeSize = [](Type type) -> unsigned {
                            if (auto outputType = type.dyn_cast<dfg::OutputType>()) {
                                type = outputType.getElementType();
                            }
                            else if (auto inputType = type.dyn_cast<dfg::InputType>()) {
                                type = inputType.getElementType();
                            }
                            
                            if (auto intType = type.dyn_cast<IntegerType>()) {
                                return intType.getWidth();
                            }
                            if (auto floatType = type.dyn_cast<FloatType>()) {
                                return floatType.getWidth();
                            }
                            return 32; // default size
                        };                    
                        // Input ports
                        for (unsigned i = 0; i < inputTypes.size(); ++i) {
                            unsigned size = getTypeSize(inputTypes[i]);
                            os << "  <Port kind=\"Input\" name=\"arg" << i << "\">\n";
                            os << "    <Type name=\"" << (inputTypes[i].isa<FloatType>() ? "float" : "int") << "\">\n";
                            os << "      <Entry kind=\"Expr\" name=\"size\">\n";
                            os << "        <Expr kind=\"Literal\" literal-kind=\"Integer\" value=\"" << size << "\"/>\n";
                            os << "      </Entry>\n";
                            os << "    </Type>\n";
                            os << "  </Port>\n";
                        }
                    
                        // Output ports
                        for (unsigned i = 0; i < outputTypes.size(); ++i) {
                            unsigned size = getTypeSize(outputTypes[i]);
                            os << "  <Port kind=\"Output\" name=\"arg" << (inputTypes.size() + i) << "\">\n";
                            os << "    <Type name=\"" << (outputTypes[i].isa<FloatType>() ? "float" : "int") << "\">\n";
                            os << "      <Entry kind=\"Expr\" name=\"size\">\n";
                            os << "        <Expr kind=\"Literal\" literal-kind=\"Integer\" value=\"" << size << "\"/>\n";
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