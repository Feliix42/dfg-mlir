//===- GenerateVitisProject.cpp - Translating to Vitis Cpp ----------------===//
//
//===----------------------------------------------------------------------===//

#include "dfg-mlir/Dialect//vitis/IR/Ops.h"
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

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
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
    LogicalResult initializeProject(StringRef funcName);
    LogicalResult createScriptFiles();
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
    std::string topFuncName;
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

static LogicalResult
printOperation(VitisProjectEmitter &emitter, ModuleOp moduleOp)
{
    // The last function in the module should be the top function
    FuncOp lastFunc =
        dyn_cast<vitis::FuncOp>(moduleOp.getBodyRegion().front().back());
    if (!lastFunc)
        return ::emitError(moduleOp.getLoc(), "Failed to find top function.");

    // Initialize the project
    if (failed(emitter.initializeProject(lastFunc.getSymName().str())))
        return ::emitError(
            moduleOp.getLoc(),
            "Failed to initialize the project.");

    for (Operation &op : moduleOp)
        if (failed(emitter.emitOperation(op))) return failure();

    // Create scripts after succesfully generate the cpp file
    if (failed(emitter.createScriptFiles())) return failure();

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
            .Case<vitis::StreamReadOp, vitis::StreamWriteOp>(
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

LogicalResult VitisProjectEmitter::initializeProject(StringRef name)
{
    // Init
    topFuncName = name.str();
    dbgOS << "Creating project for top function: " << topFuncName << "\n";
    llvm::SmallString<128> projectPath(outputDir);
    llvm::sys::path::append(projectPath, topFuncName);
    projectDir = projectPath.str().str();
    dbgOS << "Project Directory: " << projectDir << "\n";

    // Create project directory
    std::error_code ec = llvm::sys::fs::create_directory(projectDir);
    if (ec) {
        dbgOS << "Error: Failed to create directory: " << ec.message() << "\n";
        return failure();
    }
    dbgOS << "Creating project directory: " << projectDir << "\n";

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
        tclFile << "set TARGET_DEVICE \"" << targetDevice << "\"\n";
        tclFile << "set CLOCK_PERIOD 10\n";
        tclFile << "set PROJECT_DIR \"./hls_project\"\n";
        tclFile << "set SOURCE_FILE \"main.cpp\"\n";
        tclFile << "puts \"INFO: Using source file: $SOURCE_FILE\"\n";
        tclFile << "open_project $PROJECT_DIR\n";
        tclFile << "add_files $SOURCE_FILE\n";
        tclFile << "open_solution \"solution_" << topFuncName << "\"\n";
        tclFile << "set_part $TARGET_DEVICE\n";
        tclFile << "create_clock -period $CLOCK_PERIOD -name default\n";
        tclFile << "set_top " << topFuncName << "\n";
        tclFile << "csynth_design\n";
        tclFile << "export_design -rtl verilog\n";
        tclFile << "close_solution\n";
        tclFile << "close_project\n";
        tclFile << "exit\n";
        // Debug info
        dbgOS << "Created run_hls.tcl\n";
    }

    // Generate Tcl script for FPGA design
    {
        // TODO
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
        raw_indented_ostream os(shFile);
        // Create shell script to automatic run Vitis HLS and Vivado
        dbgOS << "Creating run_design.sh file\n";
        os << "#!/bin/bash\n";
        os << "if [ -z \"$XILINX_PATH\" ]; then\n";
        os.indent() << "echo \"XILINX_PATH not set\"\n";
        os << "exit 1\n";
        os.unindent() << "fi\n";
        os << "if [ -z \"$XILINX_VERSION\" ]; then\n";
        os.indent() << "echo \"XILINX_VERSION not set\"\n";
        os << "exit 1\n";
        os.unindent() << "fi\n";
        os << "VITIS_HLS=\"$XILINX_PATH/Vitis/$XILINX_VERSION/bin/"
              "vitis-run\"\n";
        os << "VIVADO=\"$XILINX_PATH/Vivado/$XILINX_VERSION/bin/vivado\"\n";
        os << "echo \"Runing Vitis HLS\"\n";
        os << "\"$VITIS_HLS\" --mode hls --tcl run_hls.tcl\n";
        os << "echo \"Runing Vivado\"\n";
        os << "\"$VIVADO\" -mode tcl -source run_vivado.tcl\n";
        os << "echo \"Successfully generate design\"\n";
        os << "DESIGN_DIR=./vivado_project\n";
        os << "BASENAME=\"" << topFuncName << "_bd\"\n";
        os << "XSA_FILENAME=\"$BASENAME.xsa\"\n";
        os << "XSA_FILE=\"$DESIGN_DIR/$XSA_FILENAME\"\n";
        os << "if [ ! -d \"$DESIGN_DIR\" ]; then\n";
        os.indent() << "echo \"Directory $DESIGN_DIR doesn't exist\"\n";
        os << "exit 1\n";
        os.unindent() << "fi\n";
        os << "if [ ! -f \"$XSA_FILE\" ]; then\n";
        os.indent() << "echo \"XSA File $XSA_FILE doesn't exist\"\n";
        os << "exit 1\n";
        os.unindent() << "fi\n";
        os << "echo \"Found XSA File: $XSA_FILE\"\n";
        os << "TEMP_DIR=\"./xsa_temp\"\n";
        os << "TARGET_DIR=\"./driver/bitfile\"\n";
        os << "echo \"Extracting \"$XSA_FILE\"\"\n";
        os << "unzip -q \"$XSA_FILE\" -d \"$TEMP_DIR\"\n";
        os << "if [ $? -ne 0 ]; then\n";
        os.indent() << "echo \"Failed to extract \"$XSA_FILE\"\"\n";
        os << "rm -rf \"$TEMP_DIR\"\n";
        os << "exit 1\n";
        os.unindent() << "fi\n";
        os << "mkdir -p \"$TARGET_DIR\"\n";
        os << "BIT_FILE=$(find \"$TEMP_DIR\" -name \"$BASENAME.bit\" | head -n "
              "1)\n";
        os << "if [ -z \"$BIT_FILE\" ]; then\n";
        os.indent() << "echo \"Failed to find \"$BASENAME.bit\"\"\n";
        os << "rm -rf \"$TEMP_DIR\"\n";
        os << "exit 1\n";
        os.unindent() << "else\n";
        os.indent() << "cp \"$BIT_FILE\" \"$TARGET_DIR\"\n";
        os << "echo \"Copied \"$BASENAME.bit\" into \"$TARGET_DIR\"\"\n";
        os.unindent() << "fi\n";
        os << "HWH_FILE=$(find \"$TEMP_DIR\" -name \"$BASENAME.hwh\" | head -n "
              "1)\n";
        os << "if [ -z \"$HWH_FILE\" ]; then\n";
        os.indent() << "echo \"Failed to find \"$BASENAME.hwh\"\"\n";
        os << "rm -rf \"$TEMP_DIR\"\n";
        os << "exit 1\n";
        os.unindent() << "else\n";
        os.indent() << "cp \"$HWH_FILE\" \"$TARGET_DIR\"\n";
        os << "echo \"Copied \"$BASENAME.hwh\" into \"$TARGET_DIR\"\"\n";
        os.unindent() << "fi\n";
        os << "rm -rf \"$TEMP_DIR\"\n";
        os << "echo \"Extract bitfile done!\"\n";
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
