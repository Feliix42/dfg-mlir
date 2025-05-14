//===- GenerateVitisProject.cpp - Translating to Vitis Cpp ----------------===//
//
//===----------------------------------------------------------------------===//

#include "dfg-mlir/Dialect//vitis/IR/Ops.h"
#include "dfg-mlir/Dialect/vitis/Enums.h"
#include "dfg-mlir/Dialect/vitis/IR/Types.h"
#include "dfg-mlir/Target/GenerateVitisProject/GenerateVitisProjectEmitter.h"
#include "mlir/IR/BlockSupport.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Path.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>

using namespace mlir;
using namespace mlir::vitis;
using llvm::formatv;

// Convenience functions to produce interleaved output with functions returning
// a LogicalResult. This is different than those in STLExtras as functions used
// on each element doesn't return a string.
template<
    typename ForwardIterator,
    typename UnaryFunctor,
    typename NullaryFunctor>
inline LogicalResult interleaveWithError(
    ForwardIterator begin,
    ForwardIterator end,
    UnaryFunctor eachFn,
    NullaryFunctor betweenFn)
{
    if (begin == end) return success();
    if (failed(eachFn(*begin))) return failure();
    ++begin;
    for (; begin != end; ++begin) {
        betweenFn();
        if (failed(eachFn(*begin))) return failure();
    }
    return success();
}

template<typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(
    const Container &c,
    UnaryFunctor eachFn,
    NullaryFunctor betweenFn)
{
    return interleaveWithError(c.begin(), c.end(), eachFn, betweenFn);
}

template<typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(
    const Container &c,
    raw_ostream &os,
    UnaryFunctor eachFn)
{
    return interleaveWithError(c.begin(), c.end(), eachFn, [&]() {
        os << ", ";
    });
}

namespace {
// Emitter class
struct VitisProjectEmitter {
    explicit VitisProjectEmitter(
        raw_ostream &os,
        std::string outputDir,
        std::string targetDevice)
            : dbgOS(os),
              outputDir(outputDir),
              targetDevice(targetDevice)
    {}

    // Class to manage argument and variable names
    struct NameManager {
        NameManager() = default;
        std::string getPrefix(Operation* op)
        {
            if (!op) return nullptr;
            auto opName = op->getName().getStringRef();

            // Get operation name without dialect name
            size_t firstDotPos = opName.find('.');
            if (firstDotPos != StringRef::npos)
                opName = opName.substr(firstDotPos + 1);

            // Search in prefix map
            auto it = prefixMap.find(opName);
            if (it != prefixMap.end()) return it->second;
            // If it's not in the map
            return "v";
        }

    private:
        const llvm::StringMap<std::string> prefixMap = {
            // ArithOps
            {     "arith.add",  "sum"},
            {    "arith.cast", "cast"},
            {     "arith.cmp",  "cmp"},
            {     "arith.div", "qout"},
            {     "arith.mul", "prod"},
            {      "arith.or",   "or"},
            {     "arith.rem",  "rem"},
            {  "arith.select",  "mux"},
            {     "arith.sub", "diff"},
            // ArrayOps
            {    "array.read", "elem"},
            {"array.ptr_read", "elem"},
            // MathOps
            {      "math.cos",  "cos"},
            {      "math.sin",  "sin"},
            // StreamOps
            {   "stream.read", "data"}
        };
    };

    // All the ops to be translated can only have one block in its region if
    // existed
    struct BlockScope {
        BlockScope(VitisProjectEmitter &emitter, Block* block)
                : emitter(emitter),
                  block(block)
        {
            emitter.dbgOS << "Entering block scope at " << block << "\n";
            // store the value-name maps before getting into this block
            oldValues = emitter.getValueNames();
        }
        ~BlockScope()
        {
            emitter.dbgOS << "Exiting block scope at " << block << "\n";
            // Remove the value-name maps added in this block
            emitter.cleanupValueNames(oldValues);
        }

    private:
        VitisProjectEmitter &emitter;
        Block* block;
        SmallVector<Value> oldValues;
    };

    // emitter methods
    LogicalResult emitOperation(Operation &op);
    LogicalResult emitType(Location loc, Type type);
    LogicalResult emitAttribute(Location loc, Attribute attr);
    LogicalResult emitVariableDeclaration(OpResult result);
    LogicalResult emitAssignPrefix(Operation &op);
    // helper functions
    raw_indented_ostream &getCppOS()
    {
        assert(cppIndentedOS && "Indented output stream not initialized");
        return *cppIndentedOS;
    }
    raw_indented_ostream &getDebugOS() { return dbgOS; }
    StringRef getTopFuncName() const { return topFuncName; }
    StringRef getProjectDir() const { return projectDir; }
    // Util functions
    LogicalResult initializeProject(
        StringRef funcName,
        size_t funcNumArgs,
        ArrayRef<int64_t> argBufferSizes,
        ArrayRef<Type> argBufferTypes,
        int64_t numInputs,
        int64_t numOutputs);
    LogicalResult createScriptFiles();
    LogicalResult createPythonDriverFiles();
    // Name manager
    bool isInScope(Value value)
    {
        return valueNames.find(value) != valueNames.end();
    }
    bool isNameInUse(const std::string &name);
    StringRef addNameForValue(Value value, const std::string &baseName);
    StringRef getOrCreateName(Value value);
    SmallVector<Value> getValueNames() const
    {
        SmallVector<Value> values;
        for (const auto &pair : valueNames) values.push_back(pair.first);
        return values;
    }
    void cleanupValueNames(const SmallVector<Value> &oldValues)
    {
        // Create a set for searching
        llvm::SmallPtrSet<Value, 16> oldValueSet;
        for (Value v : oldValues) oldValueSet.insert(v);
        // Remove all maps not in oldValues
        llvm::DenseMap<Value, std::string> newValueNames;
        for (const auto &pair : valueNames)
            if (oldValueSet.count(pair.first) > 0) newValueNames.insert(pair);
        // Restore valueNames
        valueNames = std::move(newValueNames);
    }

private:
    // Printer
    raw_indented_ostream dbgOS;
    std::unique_ptr<llvm::raw_fd_ostream> cppOS;
    std::unique_ptr<raw_indented_ostream> cppIndentedOS;
    // Command line options
    std::string outputDir;
    std::string targetDevice;
    // Utils
    std::string projectDir;
    std::string driverDir;
    std::string topFuncName;
    size_t topFuncArgSize;
    ArrayRef<int64_t> topFuncArgBufferSizes;
    ArrayRef<Type> topFuncArgBufferTypes;
    int64_t topFuncNumInputs;
    int64_t topFuncNumOutputs;
    // Helper for name generation
    llvm::DenseMap<Value, std::string> valueNames;
    NameManager nameManager;
};
} // namespace

//===----------------------------------------------------------------------===//
// Print Different Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// StandardOps
//===----------------------------------------------------------------------===//
bool reduceOneIteration = true;
std::vector<std::string> funcNames;
static LogicalResult
printOperation(VitisProjectEmitter &emitter, ModuleOp moduleOp)
{
    // The last function in the module should be the top function
    FuncOp lastFunc =
        dyn_cast<vitis::FuncOp>(moduleOp.getBodyRegion().front().back());
    if (!lastFunc)
        return ::emitError(moduleOp.getLoc(), "Failed to find top function.");

    // Initialize the project
    if (failed(emitter.initializeProject(
            lastFunc.getSymName().str(),
            lastFunc.getNumArguments(),
            cast<DenseI64ArrayAttr>(lastFunc->getAttr("argBufferSizes"))
                .asArrayRef(),
            lastFunc.getFuncElementTypes(),
            cast<IntegerAttr>(lastFunc->getAttr("num_inputs")).getInt(),
            cast<IntegerAttr>(lastFunc->getAttr("num_outputs")).getInt())))
        return ::emitError(
            moduleOp.getLoc(),
            "Failed to initialize the project.");
    

    auto &ops = moduleOp.getBody()->getOperations();        
    auto begin = ops.begin();
    auto end = ops.end();
    if (reduceOneIteration && ops.size() > 0) {
     funcNames.clear();   
        --end;

    }
    for (auto it = begin; it != end; ++it){
        Operation &op = *it;
        //
        if (failed(emitter.emitOperation(op))) return failure();
        else if (auto symbolOp = dyn_cast<mlir::SymbolOpInterface>(&op)) {
            auto funcName = symbolOp.getName().str();
            funcNames.push_back(funcName);
        }
    }
    
    for (int i = 0; i < funcNames.size(); i++) {
        llvm::errs() << "[INFO] funcName: " << funcNames[i] << "\n";
    }
    // Create scripts after succesfully generate the cpp file
    if (failed(emitter.createScriptFiles())) return failure();

    // Create python driver files
    if (failed(emitter.createPythonDriverFiles())) return failure();

    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::IncludeOp includeOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating IncludeOp at " << includeOp.getLoc() << "\n";
    cppOS << "#include \"";
    cppOS << includeOp.getInclude();
    cppOS << "\"\n";

    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::VariableOp variableOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating VariableOp at " << variableOp.getLoc() << "\n";
    if (variableOp.getInit()) {
        if (variableOp.isVariableConst()) cppOS << "const ";
        if (failed(emitter.emitAssignPrefix(*variableOp.getOperation())))
            return failure();
        if (failed(emitter.emitAttribute(
                variableOp->getLoc(),
                variableOp.getInitAttr())))
            return failure();
        cppOS << ";\n";
    } else {
        if (failed(emitter.emitVariableDeclaration(
                variableOp.getOperation()->getResult(0))))
            return failure();
        cppOS << ";\n";
    }
    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::UpdateOp updateOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating UpdateOp at " << updateOp.getLoc() << "\n";
    cppOS << emitter.getOrCreateName(updateOp.getVariable()) << " = "
          << emitter.getOrCreateName(updateOp.getNewValue()) << ";\n";
    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::FuncOp funcOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();
    VitisProjectEmitter::BlockScope scope(emitter, &funcOp.getBody().front());

    auto funcName = funcOp.getSymName().str();
    //llvm::errs() << "[INFO] funcName: " << funcName << "\n";
    dbgOS << "Translating FuncOp at " << funcOp.getLoc() << "\n";
    cppOS << "void " << funcName << "(";
    if (failed(interleaveCommaWithError(
            funcOp.getArguments(),
            cppOS,
            [&](BlockArgument arg) -> LogicalResult {
                auto argTy = arg.getType();
                if (failed(emitter.emitType(funcOp.getLoc(), argTy)))
                    return failure();
                cppOS << " ";
                if (isa<StreamType>(argTy))
                    cppOS << "&";
                else if (isa<PointerType>(argTy))
                    cppOS << "*";
                cppOS << emitter.getOrCreateName(arg);
                return success();
            })))
        return failure();
    cppOS << ")\n{\n";
    cppOS.indent();
    for (auto &opi : funcOp.getBody().getOps())
        if (failed(emitter.emitOperation(opi))) return failure();
    cppOS.unindent() << "}\n";

    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::CallOp callOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();
    // auto scope = emitter.createScope();

    dbgOS << "Translating CallOp at " << callOp.getLoc() << "\n";
    cppOS << callOp.getCallee();
    cppOS << "(";
    if (failed(interleaveCommaWithError(
            callOp.getOperands(),
            cppOS,
            [&](Value arg) -> LogicalResult {
                cppOS << emitter.getOrCreateName(arg);
                return success();
            })))
        return failure();
    cppOS << ");\n";

    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::ForOp forOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();
    VitisProjectEmitter::BlockScope scope(emitter, &forOp.getRegion().front());

    dbgOS << "Translating ForOp at " << forOp.getLoc() << "\n";
    auto loc = forOp.getLoc();
    auto lb = forOp.getLowerBound();
    auto ub = forOp.getUpperBound();
    auto step = forOp.getStep();
    auto iVar = forOp.getInductionVar();

    cppOS << "for (";
    if (failed(emitter.emitType(loc, iVar.getType()))) return failure();
    cppOS << " " << emitter.getOrCreateName(iVar) << " = " << lb << ";";
    cppOS << " " << emitter.getOrCreateName(iVar) << " < " << ub << ";";
    cppOS << " " << emitter.getOrCreateName(iVar) << " += " << step << ")";

    cppOS << " {\n";
    cppOS.indent();
    for (auto &opi : forOp.getRegion().getOps())
        if (failed(emitter.emitOperation(opi))) return failure();
    cppOS.unindent();
    cppOS << "}\n";

    return success();
}

//===----------------------------------------------------------------------===//
// ArithOps
//===----------------------------------------------------------------------===//

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::ArithAddOp addOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating ArithAddOp at " << addOp.getLoc() << "\n";
    if (failed(emitter.emitAssignPrefix(*addOp.getOperation())))
        return failure();

    cppOS << emitter.getOrCreateName(addOp.getLhs()) << " + "
          << emitter.getOrCreateName(addOp.getRhs()) << ";\n";

    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::ArithSubOp subOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating ArithSubOp at " << subOp.getLoc() << "\n";
    if (failed(emitter.emitAssignPrefix(*subOp.getOperation())))
        return failure();

    cppOS << emitter.getOrCreateName(subOp.getLhs()) << " - "
          << emitter.getOrCreateName(subOp.getRhs()) << ";\n";

    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::ArithMulOp mulOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating ArithMulOp at " << mulOp.getLoc() << "\n";
    if (failed(emitter.emitAssignPrefix(*mulOp.getOperation())))
        return failure();

    cppOS << emitter.getOrCreateName(mulOp.getLhs()) << " * "
          << emitter.getOrCreateName(mulOp.getRhs()) << ";\n";

    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::ArithDivOp divOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating ArithDivOp at " << divOp.getLoc() << "\n";
    if (failed(emitter.emitAssignPrefix(*divOp.getOperation())))
        return failure();

    cppOS << emitter.getOrCreateName(divOp.getLhs()) << " / "
          << emitter.getOrCreateName(divOp.getRhs()) << ";\n";

    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::ArithRemOp remOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating ArithRemOp at " << remOp.getLoc() << "\n";
    if (failed(emitter.emitAssignPrefix(*remOp.getOperation())))
        return failure();

    cppOS << emitter.getOrCreateName(remOp.getLhs()) << " % "
          << emitter.getOrCreateName(remOp.getRhs()) << ";\n";

    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::ArithAndOp andOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating ArithAndOp at " << andOp.getLoc() << "\n";
    if (failed(emitter.emitAssignPrefix(*andOp.getOperation())))
        return failure();

    cppOS << emitter.getOrCreateName(andOp.getLhs()) << " & "
          << emitter.getOrCreateName(andOp.getRhs()) << ";\n";

    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::ArithOrOp orOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating ArithOrOp at " << orOp.getLoc() << "\n";
    if (failed(emitter.emitAssignPrefix(*orOp.getOperation())))
        return failure();

    cppOS << emitter.getOrCreateName(orOp.getLhs()) << " | "
          << emitter.getOrCreateName(orOp.getRhs()) << ";\n";

    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::ArithCastOp castOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating ArithCastOp at " << castOp.getLoc() << "\n";
    if (failed(emitter.emitAssignPrefix(*castOp.getOperation())))
        return failure();

    cppOS << "(";
    if (failed(emitter.emitType(castOp.getLoc(), castOp.getType())))
        return failure();
    cppOS << ")";
    cppOS << emitter.getOrCreateName(castOp.getFrom()) << ";\n";

    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::ArithCmpOp cmpOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating ArithCmpOp at " << cmpOp.getLoc() << "\n";
    if (failed(emitter.emitAssignPrefix(*cmpOp.getOperation())))
        return failure();

    cppOS << emitter.getOrCreateName(cmpOp.getLhs());
    switch (cmpOp.getPredicate()) {
    case vitis::CmpPredicate::eq: cppOS << " == "; break;
    case vitis::CmpPredicate::ne: cppOS << " != "; break;
    case vitis::CmpPredicate::lt: cppOS << " < "; break;
    case vitis::CmpPredicate::le: cppOS << " <= "; break;
    case vitis::CmpPredicate::gt: cppOS << " > "; break;
    case vitis::CmpPredicate::ge: cppOS << " >= "; break;
    case vitis::CmpPredicate::three_way: cppOS << " <=> "; break;
    }
    cppOS << emitter.getOrCreateName(cmpOp.getRhs()) << ";";

    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::ArithSelectOp selectOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating ArithSelectOp at " << selectOp.getLoc() << "\n";
    if (failed(emitter.emitAssignPrefix(*selectOp.getOperation())))
        return failure();

    cppOS << emitter.getOrCreateName(selectOp.getCondition()) << " ? "
          << emitter.getOrCreateName(selectOp.getTrueValue()) << " : "
          << emitter.getOrCreateName(selectOp.getFalseValue()) << ";";

    return success();
}

//===----------------------------------------------------------------------===//
// ArrayOps
//===----------------------------------------------------------------------===//

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::ArrayReadOp readOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating ArrayReadOp at " << readOp.getLoc() << "\n";
    if (failed(emitter.emitAssignPrefix(*readOp.getOperation())))
        return failure();

    cppOS << emitter.getOrCreateName(readOp.getArray());
    for (auto idx : readOp.getIndices())
        cppOS << "[" << emitter.getOrCreateName(idx) << "]";
    cppOS << ";\n";

    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::ArrayWriteOp writeOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating ArrayWriteOp at " << writeOp.getLoc() << "\n";
    cppOS << emitter.getOrCreateName(writeOp.getArray());
    for (auto idx : writeOp.getIndices())
        cppOS << "[" << emitter.getOrCreateName(idx) << "]";
    cppOS << " = " << emitter.getOrCreateName(writeOp.getValue()) << ";\n";

    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::ArrayPointerReadOp readOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating ArrayPointerReadOp at " << readOp.getLoc() << "\n";
    if (failed(emitter.emitAssignPrefix(*readOp.getOperation())))
        return failure();

    cppOS << emitter.getOrCreateName(readOp.getArray()) << "["
          << emitter.getOrCreateName(readOp.getIndex()) << "];\n";

    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::ArrayPointerWriteOp writeOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating ArrayPointerWriteOp at " << writeOp.getLoc() << "\n";
    cppOS << emitter.getOrCreateName(writeOp.getArray()) << "["
          << emitter.getOrCreateName(writeOp.getIndex()) << "]";
    cppOS << " = " << emitter.getOrCreateName(writeOp.getValue()) << ";\n";

    return success();
}

//===----------------------------------------------------------------------===//
// MathOps
//===----------------------------------------------------------------------===//

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::MathSinOp sinOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating MathSinOp at " << sinOp.getLoc() << "\n";
    if (failed(emitter.emitAssignPrefix(*sinOp.getOperation())))
        return failure();

    cppOS << "hls::sin(" << emitter.getOrCreateName(sinOp.getValue()) << ");\n";

    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::MathCosOp cosOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating MathCosOp at " << cosOp.getLoc() << "\n";
    if (failed(emitter.emitAssignPrefix(*cosOp.getOperation())))
        return failure();

    cppOS << "hls::cos(" << emitter.getOrCreateName(cosOp.getValue()) << ");\n";

    return success();
}

//===----------------------------------------------------------------------===//
// PragmaOps
//===----------------------------------------------------------------------===//

static LogicalResult printOperation(
    VitisProjectEmitter &emitter,
    vitis::PragmaBindStorageOp bindStorageOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating PragmaBindStorageOp at " << bindStorageOp.getLoc()
          << "\n";
    cppOS << "#pragma HLS BIND_STORAGE variable="
          << emitter.getOrCreateName(bindStorageOp.getVariable());
    cppOS << " type=" << bindStorageOp.getType() << " impl=";

    auto impl = bindStorageOp.getImpl();
    if (impl == vitis::PragmaStorageImpl::automatic)
        cppOS << "auto";
    else
        cppOS << impl;
    cppOS << "\n";

    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::PragmaDataflowOp dataflowOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating PragmaDataflowOp at " << dataflowOp.getLoc() << "\n";
    cppOS << "\n#pragma HLS DATAFLOW\n";
    for (auto &opi : dataflowOp.getDataflowRegion().getOps())
        if (failed(emitter.emitOperation(opi))) return failure();

    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::PragmaInlineOp inlineOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating PragmaInlineOp at " << inlineOp.getLoc() << "\n";
    cppOS << "#pragma HLS INLINE ";
    if (inlineOp.getOff()) cppOS << "off";
    cppOS << "\n";

    return success();
}

static LogicalResult printOperation(
    VitisProjectEmitter &emitter,
    vitis::PragmaInterfaceOp interfaceOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating PragmaInterfaceOp at " << interfaceOp.getLoc()
          << "\n";
    cppOS << "#pragma HLS INTERFACE";
    cppOS << " mode=" << interfaceOp.getMode();
    cppOS << " port=" << emitter.getOrCreateName(interfaceOp.getPort());
    if (interfaceOp.getOffset()) cppOS << " offset=" << interfaceOp.getOffset();
    cppOS << " bundle=" << interfaceOp.getBundle();
    cppOS << "\n";

    return success();
}

static LogicalResult printOperation(
    VitisProjectEmitter &emitter,
    vitis::PragmaReturnInterfaceOp interfaceOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating PragmaReturnInterfaceOp at " << interfaceOp.getLoc()
          << "\n";
    cppOS << "#pragma HLS INTERFACE mode=s_axilite port=return "
             "bundle=control\n\n";

    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::PragmaPipelineOp pipelineOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating PragmaPipelineOp at " << pipelineOp.getLoc() << "\n";
    cppOS << "#pragma HLS PIPELINE II=" << pipelineOp.getInterval();
    cppOS << " style=" << pipelineOp.getStyle() << "\n";

    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::PragmaStreamOp streamOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating PragmaStreamOp at " << streamOp.getLoc() << "\n";
    cppOS << "#pragma HLS STREAM variable="
          << emitter.getOrCreateName(streamOp.getVariable());
    cppOS << " depth=" << streamOp.getDepth() << "\n";

    return success();
}

//===----------------------------------------------------------------------===//
// StreamOps
//===----------------------------------------------------------------------===//

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::StreamReadOp readOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating StreamReadOp at " << readOp.getLoc() << "\n";
    if (failed(emitter.emitAssignPrefix(*readOp.getOperation())))
        return failure();

    cppOS << emitter.getOrCreateName(readOp.getStream()) << ".read();\n";

    return success();
}

static LogicalResult
printOperation(VitisProjectEmitter &emitter, vitis::StreamWriteOp writeOp)
{
    raw_indented_ostream &cppOS = emitter.getCppOS();
    raw_indented_ostream &dbgOS = emitter.getDebugOS();

    dbgOS << "Translating StreamWriteOp at " << writeOp.getLoc() << "\n";
    cppOS << emitter.getOrCreateName(writeOp.getStream()) << ".write("
          << emitter.getOrCreateName(writeOp.getDataPkt()) << ");\n";

    return success();
}

//===----------------------------------------------------------------------===//
// Emitter Methods
//===----------------------------------------------------------------------===//

LogicalResult VitisProjectEmitter::emitOperation(Operation &op)
{
    LogicalResult status =
        llvm::TypeSwitch<Operation*, LogicalResult>(&op)
            .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
            .Case<vitis::IncludeOp, vitis::FuncOp, vitis::CallOp>(
                [&](auto op) { return printOperation(*this, op); })
            .Case<vitis::VariableOp, vitis::UpdateOp, vitis::ForOp>(
                [&](auto op) { return printOperation(*this, op); })
            .Case<
                vitis::ArithAddOp,
                vitis::ArithSubOp,
                vitis::ArithMulOp,
                vitis::ArithDivOp,
                vitis::ArithRemOp,
                vitis::ArithAndOp,
                vitis::ArithOrOp,
                vitis::ArithCastOp,
                vitis::ArithCmpOp,
                vitis::ArithSelectOp>(
                [&](auto op) { return printOperation(*this, op); })
            .Case<
                vitis::ArrayReadOp,
                vitis::ArrayWriteOp,
                vitis::ArrayPointerReadOp,
                vitis::ArrayPointerWriteOp>(
                [&](auto op) { return printOperation(*this, op); })
            .Case<vitis::MathSinOp, vitis::MathCosOp>(
                [&](auto op) { return printOperation(*this, op); })
            .Case<vitis::StreamReadOp, vitis::StreamWriteOp>(
                [&](auto op) { return printOperation(*this, op); })
            .Case<
                vitis::PragmaBindStorageOp,
                vitis::PragmaDataflowOp,
                vitis::PragmaInlineOp,
                vitis::PragmaInterfaceOp,
                vitis::PragmaReturnInterfaceOp,
                vitis::PragmaPipelineOp,
                vitis::PragmaStreamOp>(
                [&](auto op) { return printOperation(*this, op); })
            .Default(
                [](auto op) { return op->emitError("unsupported operation"); });
    return status;
}

LogicalResult VitisProjectEmitter::emitType(Location loc, Type vitisType)
{
    raw_indented_ostream &os = getCppOS();
    if (auto type = dyn_cast<IntegerType>(vitisType)) {
        auto datawidth = type.getWidth();
        if (datawidth == 1)
            return (os << "bool"), success();
        else if (type.getSignedness() == IntegerType::Unsigned)
            return (os << "ap_uint<" << datawidth << ">"), success();
        else
            return (os << "ap_int<" << datawidth << ">"), success();
    }
    if (auto type = dyn_cast<IndexType>(vitisType)) {
        os << "size_t";
        return success();
    }
    if (auto type = dyn_cast<FloatType>(vitisType)) {
        auto bitwidth = type.getIntOrFloatBitWidth();
        switch (bitwidth) {
        case 16: os << "half"; break;
        case 32: os << "float"; break;
        case 64: os << "double"; break;
        }
        return success();
    }
    if (auto type = dyn_cast<ArrayType>(vitisType)) {
        if (failed(emitType(loc, type.getElementType()))) return failure();
        return success();
    }
    if (auto type = dyn_cast<APFixedType>(vitisType)) {
        os << "ap_fixed<" << type.getDatawidth() << ", " << type.getIntWidth()
           << ">";
        return success();
    }
    if (auto type = dyn_cast<APFixedUType>(vitisType)) {
        os << "ap_ufixed<" << type.getDatawidth() << ", " << type.getIntWidth()
           << ">";
        return success();
    }
    if (auto type = dyn_cast<StreamType>(vitisType)) {
        os << "hls::stream<";
        if (failed(emitType(loc, type.getStreamType()))) return failure();
        os << ">";
        return success();
    }
    if (auto type = dyn_cast<PointerType>(vitisType)) {
        if (failed(emitType(loc, type.getPointerType()))) return failure();
        return success();
    }

    return emitError(loc, "cannot emit type ") << vitisType;
}

LogicalResult VitisProjectEmitter::emitAttribute(Location loc, Attribute attr)
{
    raw_indented_ostream &os = getCppOS();
    auto printInt = [&](const APInt &val, bool isUnsigned) {
        if (val.getBitWidth() == 1) {
            if (val.getBoolValue())
                os << "true";
            else
                os << "false";
        } else {
            SmallString<128> strValue;
            val.toString(strValue, 10, !isUnsigned, false);
            os << strValue;
        }
    };
    auto printFloat = [&](const APFloat &val) {
        if (val.isFinite()) {
            SmallString<128> strValue;
            // Use default values of toString except don't truncate zeros.
            val.toString(strValue, 0, 0, false);
            os << strValue;
        } else if (val.isNaN()) {
            os << "NAN";
        } else if (val.isInfinity()) {
            if (val.isNegative()) os << "-";
            os << "INFINITY";
        }
    };

    // Print integer attributes.
    if (auto iAttr = dyn_cast<IntegerAttr>(attr)) {
        if (auto iType = dyn_cast<IntegerType>(iAttr.getType())) {
            printInt(
                iAttr.getValue(),
                iType.getSignedness() == IntegerType::Unsigned);
            return success();
        }
        if (auto iType = dyn_cast<IndexType>(iAttr.getType())) {
            printInt(iAttr.getValue(), false);
            return success();
        }
    }
    if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
        if (auto iType = dyn_cast<IntegerType>(dense.getElementType())) {
            os << '{';
            if (dense.isSplat()) {
                printInt(
                    *dense.getValues<APInt>().begin(),
                    iType.getSignedness() == IntegerType::Unsigned);
            } else {
                interleaveComma(dense, os, [&](const APInt &val) {
                    printInt(
                        val,
                        iType.getSignedness() == IntegerType::Unsigned);
                });
            }
            os << '}';
            return success();
        }
        if (auto iType = dyn_cast<IndexType>(dense.getElementType())) {
            os << '{';
            if (dense.isSplat()) {
                printInt(*dense.getValues<APInt>().begin(), false);
            } else {
                interleaveComma(dense, os, [&](const APInt &val) {
                    printInt(val, false);
                });
            }
            os << '}';
            return success();
        }
    }

    // Print floating point attributes.
    if (auto fAttr = dyn_cast<FloatAttr>(attr)) {
        if (!isa<Float16Type, Float32Type, Float64Type>(fAttr.getType())) {
            return emitError(
                loc,
                "expected floating point attribute to be f16, f32 or "
                "f64");
        }
        printFloat(fAttr.getValue());
        return success();
    }
    if (auto dense = dyn_cast<DenseFPElementsAttr>(attr)) {
        if (!isa<Float16Type, Float32Type, Float64Type>(
                dense.getElementType())) {
            return emitError(
                loc,
                "expected floating point attribute to be f16, f32 or "
                "f64");
        }
        os << '{';
        if (dense.isSplat()) {
            printFloat(*dense.getValues<APFloat>().begin());
        } else {
            interleaveComma(dense, os, [&](const APFloat &val) {
                printFloat(val);
            });
        }
        os << '}';
        return success();
    }

    // Print symbolic reference attributes.
    if (auto sAttr = dyn_cast<SymbolRefAttr>(attr)) {
        if (sAttr.getNestedReferences().size() > 1)
            return emitError(loc, "attribute has more than 1 nested reference");
        os << sAttr.getRootReference().getValue();
        return success();
    }

    // Print type attributes.
    if (auto type = dyn_cast<TypeAttr>(attr))
        return emitType(loc, type.getValue());

    return emitError(loc, "cannot emit attribute: ") << attr;
}

LogicalResult VitisProjectEmitter::emitVariableDeclaration(OpResult result)
{
    raw_indented_ostream &os = getCppOS();
    if (isInScope(result)) {
        return result.getDefiningOp()->emitError(
            "result variable for the operation already declared");
    }
    if (failed(emitType(result.getOwner()->getLoc(), result.getType())))
        return failure();
    os << " " << getOrCreateName(result);
    if (auto arrayTy = dyn_cast<ArrayType>(result.getType()))
        for (auto dim : arrayTy.getShape()) os << "[" << dim << "]";
    return success();
}

LogicalResult VitisProjectEmitter::emitAssignPrefix(Operation &op)
{
    raw_indented_ostream &os = getCppOS();
    switch (op.getNumResults()) {
    case 1:
    {
        OpResult result = op.getResult(0);
        if (failed(emitVariableDeclaration(result))) return failure();
        os << " = ";
        break;
    }
    default: return failure();
    }
    return success();
}

LogicalResult VitisProjectEmitter::initializeProject(
    StringRef funcName,
    size_t funcNumArgs,
    ArrayRef<int64_t> argBufferSizes,
    ArrayRef<Type> argBufferTypes,
    int64_t numInputs,
    int64_t numOutputs)
{
    // Init
    topFuncName = funcName.str();
    topFuncArgSize = funcNumArgs;
    topFuncArgBufferSizes = argBufferSizes;
    topFuncArgBufferTypes = argBufferTypes;
    topFuncNumInputs = numInputs;
    topFuncNumOutputs = numOutputs;
    dbgOS << "Creating project for top function: " << topFuncName << "\n";
    llvm::SmallString<128> projectPath(outputDir);
    llvm::sys::path::append(projectPath, topFuncName);
    projectDir = projectPath.str().str();
    dbgOS << "Project Directory: " << projectDir << "\n";

    // Create project directory
    dbgOS << "Creating project directory: " << projectDir << "\n";
    std::error_code ec = llvm::sys::fs::create_directory(projectPath);
    if (ec) {
        dbgOS << "Error: Failed to create directory: " << ec.message() << "\n";
        return failure();
    }
    dbgOS << "Created project directory\n";
    // Create driver directory inside project dir
    llvm::SmallString<128> driverPath(projectDir);
    llvm::sys::path::append(driverPath, "driver/driver");
    driverDir = driverPath.str();
    dbgOS << "Creating driver directory: " << driverDir << "\n";
    ec = llvm::sys::fs::create_directories(driverPath);
    if (ec) {
        dbgOS << "Error: Failed to create directory: " << ec.message() << "\n";
        return failure();
    }
    dbgOS << "Created driver directory\n";

    // Create main.cpp
    SmallString<128> cppFilePath(projectDir);
    llvm::sys::path::append(cppFilePath, "main.cpp");
    ec = std::error_code();
    cppOS = std::make_unique<llvm::raw_fd_ostream>(
        cppFilePath.str(),
        ec,
        llvm::sys::fs::OF_Text);
    if (ec) {
        dbgOS << "Error: Failed to create main.cpp: " << ec.message() << "\n";
        return failure();
    }
    cppIndentedOS = std::make_unique<raw_indented_ostream>(*cppOS);

    dbgOS << "Creating main.cpp file\n";
    *cppOS << "// Generated HLS code from MLIR Vitis Dialect\n";

    return success();
}

namespace {
std::string
replaceAll(std::string str, const std::string &from, const std::string &to)
{
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return str;
}
constexpr const char* kRunHLSTclTemplate =
    R"(set TARGET_DEVICE "{{TARGET_DEVICE}}"
set CLOCK_PERIOD 10
set PROJECT_DIR "./hls_project"
set SOURCE_FILE "main.cpp"
if {![file exists $SOURCE_FILE]} {
    puts "ERROR: Source file $SOURCE_FILE does not exist!"
    exit 1
}
puts "INFO: Using source file: $SOURCE_FILE"
open_project $PROJECT_DIR
add_files $SOURCE_FILE
puts "INFO: Creating HLS solution"
open_solution "solution_{{TOP_FUNC_NAME}}"
set_part $TARGET_DEVICE
create_clock -period $CLOCK_PERIOD -name default
set_top {{TOP_FUNC_NAME}}
puts "INFO: Set top function to '{{TOP_FUNC_NAME}}'"
puts "INFO: Running C synthesis..."
if {[catch {csynth_design} result]} {
    puts "ERROR: C synthesis failed: $result"
    exit 1
}
puts "INFO: C synthesis completed successfully"
puts "INFO: Exporting RTL design..."
if {[catch {export_design -rtl verilog} result]} {
    puts "ERROR: Failed to export design: $result"
    exit 1
}
puts "INFO: Design exported successfully"
close_solution
close_project
puts "INFO: HLS completed successfully"
exit
)";
constexpr const char* kRunVivadoTclTemplate =
    R"(set project_name "{{TOP_FUNC_NAME}}"
set project_dir "./vivado_project/"
set project_path "$project_dir/$project_name.xpr"
set target_device "{{TARGET_DEVICE}}"
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
create_bd_cell -type ip -vlnv xilinx.com:hls:{{TOP_FUNC_NAME}}:1.0 {{TOP_FUNC_NAME}}
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/zynq_mpsoc/M_AXI_HPM0_FPD} Slave {/{{TOP_FUNC_NAME}}/s_axi_control} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}}  [get_bd_intf_pins {{TOP_FUNC_NAME}}/s_axi_control]
{{GMEM_AXI_CONNECTIONS}}puts "INFO: Validating block design: $bd_name"
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
)";

constexpr const char* kRunDesignShTemplate =
    R"(#!/bin/bash
if [ -z "$XILINX_PATH" ]; then
    echo "XILINX_PATH not set"
    exit 1
fi
if [ -z "$XILINX_VERSION" ]; then
    echo "XILINX_VERSION not set"
    exit 1
fi
VITIS_HLS="$XILINX_PATH/Vitis/$XILINX_VERSION/bin/vitis-run"
VIVADO="$XILINX_PATH/Vivado/$XILINX_VERSION/bin/vivado"
echo "Runing Vitis HLS"
"$VITIS_HLS" --mode hls --tcl run_hls.tcl
if [ $? -ne 0 ]; then
    echo "ERROR: Vitis HLS execution failed"
    exit 1
fi
echo "Runing Vivado"
"$VIVADO" -mode tcl -source run_vivado.tcl
if [ $? -ne 0 ]; then
    echo "ERROR: Vivado execution failed"
    exit 1
fi
echo "Successfully generate design"
DESIGN_DIR=./vivado_project
BASENAME="{{TOP_FUNC_NAME}}_bd"
XSA_FILENAME="$BASENAME.xsa"
XSA_FILE="$DESIGN_DIR/$XSA_FILENAME"
if [ ! -d "$DESIGN_DIR" ]; then
    echo "Directory $DESIGN_DIR doesn't exist"
    exit 1
fi
if [ ! -f "$XSA_FILE" ]; then
    echo "XSA File $XSA_FILE doesn't exist"
    exit 1
fi
echo "Found XSA File: $XSA_FILE"
TEMP_DIR="./xsa_temp"
TARGET_DIR="./driver/bitfile"
echo "Extracting "$XSA_FILE""
unzip -q "$XSA_FILE" -d "$TEMP_DIR"
if [ $? -ne 0 ]; then
    echo "Failed to extract "$XSA_FILE""
    rm -rf "$TEMP_DIR"
    exit 1
fi
mkdir -p "$TARGET_DIR"
BIT_FILE=$(find "$TEMP_DIR" -name "$BASENAME.bit" | head -n 1)
if [ -z "$BIT_FILE" ]; then
    echo "Failed to find "$BASENAME.bit""
    rm -rf "$TEMP_DIR"
    exit 1
else
    cp "$BIT_FILE" "$TARGET_DIR"
    echo "Copied "$BASENAME.bit" into "$TARGET_DIR""
fi
HWH_FILE=$(find "$TEMP_DIR" -name "$BASENAME.hwh" | head -n 1)
if [ -z "$HWH_FILE" ]; then
    echo "Failed to find "$BASENAME.hwh""
    rm -rf "$TEMP_DIR"
    exit 1
else
    cp "$HWH_FILE" "$TARGET_DIR"
    echo "Copied "$BASENAME.hwh" into "$TARGET_DIR""
fi
rm -rf "$TEMP_DIR"
echo "Extract bitfile done!"
)";
} // namespace

LogicalResult VitisProjectEmitter::createScriptFiles()
{
    // Generate Tcl script for HLS design
    {
        SmallString<128> tclPath(projectDir);
        llvm::sys::path::append(tclPath, "run_hls.tcl");
        std::error_code ec;
        llvm::raw_fd_ostream tclFile(tclPath, ec, llvm::sys::fs::OF_Text);
        if (ec) {
            dbgOS << "Error creating run_hls.tcl: " << ec.message() << "\n";
            return failure();
        }
        
        // Create tcl script based on top function name and target device
        dbgOS << "Creating run_hls.tcl file\n";
        // Use pre-defined template and replace the key
        std::string content;
                if (!reduceOneIteration) {
            // Normal case: single top function
            std::string tempcontent(kRunHLSTclTemplate);
            tempcontent = replaceAll(tempcontent, "{{TARGET_DEVICE}}", targetDevice);
            tempcontent = replaceAll(tempcontent, "{{TOP_FUNC_NAME}}", topFuncName);
            content = tempcontent;
        } else {
            std::string templateStr(kRunHLSTclTemplate);
            const std::string loopStart = "open_solution \"solution_{{TOP_FUNC_NAME}}\"";
            const std::string loopEnd = "close_solution";
            size_t startPos = templateStr.find(loopStart);
            size_t endPos = templateStr.find(loopEnd, startPos);
            if (startPos == std::string::npos || endPos == std::string::npos) {
                dbgOS << "Error: Template does not contain expected block\n";
                return failure();
            }
            endPos += std::string("close_solution").length();
            std::string prefix = templateStr.substr(0, startPos);
            std::string loopBlock = templateStr.substr(startPos, endPos - startPos);
            std::string suffix = templateStr.substr(endPos);
            prefix = replaceAll(prefix, "{{TARGET_DEVICE}}", targetDevice);
            content += prefix;
            for (const auto &funcName : funcNames) {
                std::string block = loopBlock;
                block = replaceAll(block, "{{TOP_FUNC_NAME}}", funcName);
                content += block + "\n";
            }
            content += suffix;
        }
        
        tclFile << content;
        // Debug info
        dbgOS << "Created run_hls.tcl\n";
    }
     
    // Generate Tcl script for FPGA design
    if(!reduceOneIteration)
    {
        SmallString<128> tclPath(projectDir);
        llvm::sys::path::append(tclPath, "run_vivado.tcl");
        std::error_code ec;
        llvm::raw_fd_ostream tclFile(tclPath, ec, llvm::sys::fs::OF_Text);
        if (ec) {
            dbgOS << "Error creating run_vivado.tcl: " << ec.message() << "\n";
            return failure();
        }
        // Create tcl script based on top function name and target device
        dbgOS << "Creating run_vivado.tcl file\n";
        // Use pre-defined template and replace the key
        std::string gmemConnections;
        // Add automation for m_axi ports
        if (topFuncArgSize != 0) {
            std::string configAutomation =
                "apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { ";
            std::string configClock =
                "Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} ";
            std::string configMAxi =
                "Master {/" + topFuncName + "/m_axi_gmem_arg0} ";
            std::string configSAxi =
                "Slave {/zynq_mpsoc/S_AXI_HPC0_FPD} ddr_seg {Auto} ";
            std::string configConnect =
                "intc_ip {New AXI SmartConnect} master_apm {0}} "
                "[get_bd_intf_pins zynq_mpsoc/S_AXI_HPC0_FPD]\n";
            gmemConnections.append(
                configAutomation + configClock + configMAxi + configSAxi
                + configConnect);
            // Already created smart connect ip
            for (size_t i = 1; i < topFuncArgSize; i++) {
                configAutomation =
                    "apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config "
                    "{ ";
                configClock =
                    "Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} ";
                configMAxi = "Master {/" + topFuncName + "/m_axi_gmem_arg"
                             + std::to_string(i) + "} ";
                configSAxi =
                    "Slave {/zynq_mpsoc/S_AXI_HPC0_FPD} ddr_seg {Auto} ";
                configConnect =
                    "intc_ip {/axi_smc_1} master_apm {0}} [get_bd_intf_pins "
                    + topFuncName + "/m_axi_gmem_arg" + std::to_string(i)
                    + "]\n";
                gmemConnections.append(
                    configAutomation + configClock + configMAxi + configSAxi
                    + configConnect);
            }
        }
        std::string content(kRunVivadoTclTemplate);
        content = replaceAll(content, "{{TOP_FUNC_NAME}}", topFuncName);
        content = replaceAll(content, "{{TARGET_DEVICE}}", targetDevice);
        content =
            replaceAll(content, "{{GMEM_AXI_CONNECTIONS}}", gmemConnections);
        tclFile << content;
        // Debug info
        dbgOS << "Created run_vivado.tcl\n";
    }

    // Genreate Shell script for auto control
    {
        SmallString<128> shPath(projectDir);
        llvm::sys::path::append(shPath, "run_design.sh");
        std::error_code ec;
        llvm::raw_fd_ostream shFile(shPath, ec, llvm::sys::fs::OF_Text);
        if (ec) {
            dbgOS << "Error creating run_design.sh: " << ec.message() << "\n";
            return failure();
        }
        // Create shell script to automatic run Vitis HLS and Vivado
        dbgOS << "Creating run_design.sh file\n";
        std::string content(kRunDesignShTemplate);
        if (reduceOneIteration) {
            std::string vivadoStart = R"(echo "Runing Vivado")";
            size_t pos = content.find(vivadoStart);
            if (pos != std::string::npos) {
                content = content.substr(0, pos); // Truncate everything from Vivado onwards
            }
        }
       
        content = replaceAll(content, "{{TOP_FUNC_NAME}}", topFuncName);
        shFile << content;
        // Debug info
        dbgOS << "Created run_design.sh\n";
        // Set up permissions
        llvm::sys::fs::perms perm = static_cast<llvm::sys::fs::perms>(
            llvm::sys::fs::all_read | llvm::sys::fs::all_write
            | llvm::sys::fs::all_exe);
        ec = llvm::sys::fs::setPermissions(shPath, perm);
        if (ec) {
            dbgOS << "Warning: Could not set permissions on run_design.sh: "
                  << ec.message() << "\n";
        }
    }

    return success();
}

namespace {
constexpr const char* kPythonDriverInit =
    R"(from .driver import Accelerator
__all__ = ['Accelerator']
)";
constexpr const char* kPythonDriver = R"(import numpy as np
import time
from pynq import PL, allocate, Overlay
class Accelerator:
    def __init__(self):
        PL.reset()
        self.ol = Overlay("bitfile/{{TOP_FUNC_NAME}}_bd.bit")
        self.ip = self.ol.{{TOP_FUNC_NAME}}
        self.buffer_sizes = {{TOP_FUNC_ARG_BUFFER_SIZES}}
        self.buffer_dtypes = {{TOP_FUNC_ARG_BUFFER_TYPES}}
        self.num_inputs = {{TOP_FUNC_NUM_INPUTS}}
        self.num_outputs = {{TOP_FUNC_NUM_OUTPUTS}}
        self.num_args = self.num_inputs + self.num_outputs
        self.buffers = []
        for i in range(self.num_args):
            self.buffers.append(allocate(shape=(self.buffer_sizes[i],), dtype=self.buffer_dtypes[i]))
        self._configure_addr()
        self.exec_time = None
        print(f"Initialize succesfully, please give {self.num_inputs} inputs with sizes {self.buffer_sizes[:self.num_inputs]} and type {self.buffer_dtypes[:self.num_inputs]}" + " or their multiple")
    def _configure_addr(self):
        for i in range(self.num_args):
            addr = self.buffers[i].physical_address
            setattr(self.ip.register_map, f"arg{i}_1", addr & 0xFFFFFFFF)
            setattr(self.ip.register_map, f"arg{i}_2", (addr >> 32) & 0xFFFFFFFF)
    def _run_ip(self):
        self.ip.register_map.CTRL.AP_START = 1
        while self.ip.register_map.CTRL.AP_DONE == 0:
            pass
    def compute(self, inputs):
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}")
        num_iters = []
        for i, e in enumerate(inputs):
            input_size = e.size
            buffer_size = self.buffer_sizes[i]
            if input_size % buffer_size != 0:
                raise ValueError(f"Input {i} size ({input_size}) must be a multiple of {self.buffer_size}")
            num_iters.append(input_size // buffer_size)
        if len(set(num_iters)) > 1:
            raise ValueError(f"All inputs must have sizes that result in the same number of iterations. Got: {num_iters}")
        num_iter = num_iters[0]
        print(f"The kernel will be executed {num_iter} times")
        start_time = time.time()
        input_blocks = []
        for i, inp in enumerate(inputs):
            input_blocks.append(inp.reshape(num_iter, self.buffer_sizes[i]))
        output_blocks = []
        for i in range(self.num_outputs):
            idx = i + self.num_inputs
            output_blocks.append(np.zeros((num_iter, self.buffer_sizes[idx]), dtype=self.buffer_dtypes[idx]))
        for i in range(num_iter):
            for j in range(self.num_inputs):
                np.copyto(self.buffers[j], input_blocks[j][i])
            self._run_ip()
            for j in range(self.num_outputs):
                idx = j + self.num_inputs
                output_blocks[j][i, :] = self.buffers[idx]
        self.exec_time = time.time() - start_time
        outputs = []
        for block in output_blocks:
            outputs.append(block.reshape(-1))
        return outputs
    def get_execution_time(self):
        if self.exec_time is None:
            return None
        return self.exec_time
)";
} // namespace

static std::string typeToNumpyTypeString(Type type)
{
    // Integers
    if (auto intType = dyn_cast<IntegerType>(type)) {
        unsigned width = intType.getWidth();

        // Boolean type
        if (width == 1) return "bool";

        bool isSigned = !intType.isUnsigned();
        // promote ap_int to nearest integer
        unsigned standardWidth =
            1 << static_cast<unsigned>(std::ceil(std::log2(width)));
        if (standardWidth < 8 || standardWidth > 64) return "unknown";
        // Fix signedness
        std::string prefix = isSigned ? "int" : "uint";
        return prefix + std::to_string(standardWidth);
    }
    // Floating point numbers
    if (auto floatType = dyn_cast<FloatType>(type)) {
        unsigned width = floatType.getWidth();
        if (width == 16)
            return "float16";
        else if (width == 32)
            return "float32";
        else if (width == 64)
            return "float64";
        else
            return "unknown";
    }
    // Default
    return "unknown";
}

LogicalResult VitisProjectEmitter::createPythonDriverFiles()
{
    auto sizeArrayToString = [&](llvm::ArrayRef<int64_t> array) -> std::string {
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < array.size(); ++i) {
            ss << array[i];
            if (i < array.size() - 1) ss << ", ";
        }
        ss << "]";
        return ss.str();
    };
    auto typeArrayToString = [&](ArrayRef<Type> array) -> std::string {
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < array.size(); ++i) {
            ss << "np.";
            ss << typeToNumpyTypeString(array[i]);
            if (i < array.size() - 1) ss << ", ";
        }
        ss << "]";
        return ss.str();
    };
    {
        SmallString<128> initPath(driverDir);
        llvm::sys::path::append(initPath, "__init__.py");
        std::error_code ec;
        llvm::raw_fd_ostream initFile(initPath, ec, llvm::sys::fs::OF_Text);
        if (ec) {
            dbgOS << "Error creating __init__.py: " << ec.message() << "\n";
            return failure();
        }
        // Create Python driver init file
        dbgOS << "Creating __init__.py file\n";
        std::string content(kPythonDriverInit);
        initFile << content;
        // Debug info
        dbgOS << "Created __init__.py\n";
    }
    {
        SmallString<128> driverPath(driverDir);
        llvm::sys::path::append(driverPath, "driver.py");
        std::error_code ec;
        llvm::raw_fd_ostream driverFile(driverPath, ec, llvm::sys::fs::OF_Text);
        if (ec) {
            dbgOS << "Error creating driver.py: " << ec.message() << "\n";
            return failure();
        }
        // Create Python driver init file
        dbgOS << "Creating driver.py file\n";
        std::string content(kPythonDriver);
        content = replaceAll(content, "{{TOP_FUNC_NAME}}", topFuncName);
        content = replaceAll(
            content,
            "{{TOP_FUNC_ARG_BUFFER_SIZES}}",
            sizeArrayToString(topFuncArgBufferSizes));
        content = replaceAll(
            content,
            "{{TOP_FUNC_ARG_BUFFER_TYPES}}",
            typeArrayToString(topFuncArgBufferTypes));
        content = replaceAll(
            content,
            "{{TOP_FUNC_NUM_INPUTS}}",
            std ::to_string(topFuncNumInputs));
        content = replaceAll(
            content,
            "{{TOP_FUNC_NUM_OUTPUTS}}",
            std ::to_string(topFuncNumOutputs));
        driverFile << content;
        // Debug info
        dbgOS << "Created driver.py\n";
    }
    return success();
}

bool VitisProjectEmitter::isNameInUse(const std::string &name)
{
    for (const auto &entry : valueNames)
        if (entry.second == name) return true;
    return false;
}

StringRef
VitisProjectEmitter::addNameForValue(Value value, const std::string &baseName)
{
    unsigned suffix = 0;
    std::string nameStr;
    do {
        nameStr = formatv("{0}{1}", baseName, suffix++);
    } while (isNameInUse(nameStr));
    valueNames[value] = nameStr;

    dbgOS << "Created name " << nameStr << " for " << value << " at "
          << value.getLoc() << "\n";
    return valueNames[value];
}

StringRef VitisProjectEmitter::getOrCreateName(Value value)
{
    // Check if the value already has a name
    auto it = valueNames.find(value);
    if (it != valueNames.end()) {
        dbgOS << "Found name " << valueNames[value] << " for " << value
              << " at " << value.getLoc() << "\n";
        return it->second;
    }

    std::string name;
    Operation* definingOp = value.getDefiningOp();

    // Treat FuncOp and ForOp block argument differently
    // If there is no defining operation, it's a block argument
    if (!definingOp) {
        if (auto blkArg = dyn_cast<BlockArgument>(value)) {
            Block* parentBlock = blkArg.getParentBlock();
            Operation* parentOp = parentBlock->getParentOp();

            if (parentOp) {
                if (auto funcOp = dyn_cast<FuncOp>(parentOp))
                    return addNameForValue(value, "arg");
                else if (auto forOp = dyn_cast<ForOp>(parentOp))
                    return addNameForValue(value, "idx");
                else
                    return addNameForValue(value, "v");
            }
        }
    }
    // Otherwise, it's defined by another operation
    else {
        std::string prefix;
        if (auto varOp = dyn_cast<VariableOp>(definingOp)) {
            if (varOp.isVariableConst())
                prefix = "cst";
            else if (isa<ArrayType>(varOp.getType()))
                prefix = "array";
            else if (isa<StreamType>(varOp.getType()))
                prefix = "stream";
            else
                prefix = "var";
        } else {
            prefix = nameManager.getPrefix(definingOp);
        }
        if (!isa<FuncOp>(value.getParentBlock()->getParentOp()))
            prefix += "_tmp";
        return addNameForValue(value, prefix);
    }
}

//===----------------------------------------------------------------------===//
// Generate Vitis Project
//===----------------------------------------------------------------------===//

LogicalResult vitis::generateVitisProject(
    Operation* op,
    raw_ostream &os,
    StringRef outputDir,
    StringRef targetDevice)
{
    VitisProjectEmitter emitter(os, outputDir.str(), targetDevice.str());
    return emitter.emitOperation(*op);
}
