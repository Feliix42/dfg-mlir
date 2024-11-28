//===- TranslateToVitisCpp.cpp - Translating to Vitis Cpp -----------------===//
//
//===----------------------------------------------------------------------===//

#include "dfg-mlir/Dialect//vitis/IR/Ops.h"
#include "dfg-mlir/Dialect/vitis/IR/Types.h"
#include "dfg-mlir/Target/VitisCpp/VitisCppEmitter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/IndentedOstream.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include <llvm/Support/Casting.h>
#include <mlir/IR/BlockSupport.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LogicalResult.h>
#include <stack>

#define DEBUG_TYPE "translate-to-vitiscpp"

using namespace mlir;
using namespace mlir::vitis;
using llvm::formatv;

/// Convenience functions to produce interleaved output with functions returning
/// a LogicalResult. This is different than those in STLExtras as functions used
/// on each element doesn't return a string.
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

struct VitisCppEmitter {
    explicit VitisCppEmitter(raw_ostream &os) : os(os)
    {
        valueInScopeCount.push(0);
        labelInScopeCount.push(0);
    }

    LogicalResult emitOperation(Operation &op, bool trailingSemicolon);
    LogicalResult emitType(Location loc, Type type);
    LogicalResult emitTypes(Location loc, ArrayRef<Type> types);
    LogicalResult emitTupleType(Location loc, ArrayRef<Type> types);
    LogicalResult emitAttribute(Location loc, Attribute attr);
    LogicalResult emitAssignPrefix(Operation &op);
    LogicalResult
    emitVariableDeclaration(OpResult result, bool trailingSemicolon);

    StringRef getOrCreateName(Value value);
    StringRef getOrCreateName(Block &block);

    raw_indented_ostream &ostream() { return os; }
    bool hasValueInScope(Value val) { return valueMapper.count(val); }

    struct Scope {
        Scope(VitisCppEmitter &emitter)
                : valueMapperScope(emitter.valueMapper),
                  blockMapperScope(emitter.blockMapper),
                  emitter(emitter)
        {
            emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
            emitter.labelInScopeCount.push(emitter.labelInScopeCount.top());
        }
        ~Scope()
        {
            emitter.valueInScopeCount.pop();
            emitter.labelInScopeCount.pop();
        }

    private:
        llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
        llvm::ScopedHashTableScope<Block*, std::string> blockMapperScope;
        VitisCppEmitter &emitter;
    };

private:
    using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
    using BlockMapper = llvm::ScopedHashTable<Block*, std::string>;
    raw_indented_ostream os;

    ValueMapper valueMapper;
    BlockMapper blockMapper;

    std::stack<int64_t> valueInScopeCount;
    std::stack<int64_t> labelInScopeCount;
};

} // namespace

/// Return the existing or a new name for a Value.
StringRef VitisCppEmitter::getOrCreateName(Value value)
{
    if (!valueMapper.count(value))
        valueMapper.insert(value, formatv("v{0}", ++valueInScopeCount.top()));
    return *valueMapper.begin(value);
}

/// Return the existing or a new label for a Block.
StringRef VitisCppEmitter::getOrCreateName(Block &block)
{
    if (!blockMapper.count(&block))
        blockMapper.insert(
            &block,
            formatv("label{0}", ++labelInScopeCount.top()));
    return *blockMapper.begin(&block);
}

LogicalResult VitisCppEmitter::emitAttribute(Location loc, Attribute attr)
{
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
        os << ";";
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

LogicalResult VitisCppEmitter::emitVariableDeclaration(
    OpResult result,
    bool trailingSemicolon)
{
    if (hasValueInScope(result)) {
        return result.getDefiningOp()->emitError(
            "result variable for the operation already declared");
    }
    if (failed(emitType(result.getOwner()->getLoc(), result.getType())))
        return failure();
    os << " " << getOrCreateName(result);
    if (trailingSemicolon) os << ";\n";
    return success();
}

LogicalResult VitisCppEmitter::emitAssignPrefix(Operation &op)
{
    switch (op.getNumResults()) {
    case 0: break;
    case 1:
    {
        OpResult result = op.getResult(0);
        if (failed(emitVariableDeclaration(result, false))) return failure();
        os << " = ";
        break;
    }
    default:
        os << "std::tie(";
        interleaveComma(op.getResults(), os, [&](Value result) {
            os << getOrCreateName(result);
        });
        os << ") = ";
    }
    return success();
}

//===----------------------------------------------------------------------===//
// StandardOps
//===----------------------------------------------------------------------===//

static LogicalResult printOperation(VitisCppEmitter &emitter, ModuleOp moduleOp)
{
    VitisCppEmitter::Scope scope(emitter);
    raw_indented_ostream &os = emitter.ostream();

    os << "#include\"ap_axi_sdata.h\"\n"
       << "#include\"ap_int.h\"\n"
       << "#include\"hls_stream.h\"\n\n";

    os << "extern \"C\" {\n";
    for (Operation &op : moduleOp)
        if (failed(emitter.emitOperation(op, false))) return failure();
    os << "}";

    return success();
}

static LogicalResult
printOperation(VitisCppEmitter &emitter, vitis::FuncOp funcOp)
{
    VitisCppEmitter::Scope scope(emitter);
    raw_indented_ostream &os = emitter.ostream();

    if (failed(emitter.emitTypes(
            funcOp.getLoc(),
            funcOp.getFunctionType().getResults())))
        return failure();
    os << " " << funcOp.getSymName() << "(";
    if (failed(interleaveCommaWithError(
            funcOp.getArguments(),
            os,
            [&](BlockArgument arg) -> LogicalResult {
                if (failed(emitter.emitType(funcOp.getLoc(), arg.getType())))
                    return failure();
                os << " &" << emitter.getOrCreateName(arg);
                return success();
            })))
        return failure();
    os << ") {\n";
    for (size_t i = 0; i < funcOp.getNumArguments(); i++)
        os << "#pragma HLS INTERFACE mode=axis port=v" << i + 1 << "\n";
    os << "#pragma HLS INTERFACE mode=s_axilite port=return bundle=control\n";
    os.indent();

    Region::BlockListType &blocks = funcOp.getBlocks();
    for (Block &block : blocks) emitter.getOrCreateName(block);

    for (auto &opi : funcOp.getBody().getOps())
        if (failed(emitter.emitOperation(opi, false))) return failure();

    os.unindent() << "}";
    return success();
}

// TODO: Curretnly only support void return, deal with this later
static LogicalResult
printOperation(VitisCppEmitter &emitter, vitis::ReturnOp returnOp)
{
    return success();
}

static LogicalResult
printOperation(VitisCppEmitter &emitter, vitis::ConstantOp constantOp)
{
    if (failed(emitter.emitAssignPrefix(*constantOp.getOperation())))
        return failure();
    return emitter.emitAttribute(constantOp->getLoc(), constantOp.getValue());
}

static LogicalResult
printOperation(VitisCppEmitter &emitter, vitis::DefineOp defineOp)
{
    if (failed(emitter.emitVariableDeclaration(
            defineOp.getOperation()->getResult(0),
            true)))
        return failure();
    return success();
}

static LogicalResult
printOperation(VitisCppEmitter &emitter, vitis::UpdateOp updateOp)
{
    raw_indented_ostream &os = emitter.ostream();

    os << emitter.getOrCreateName(updateOp.getVariable()) << " = "
       << emitter.getOrCreateName(updateOp.getNewValue()) << ";";
    return success();
}

static LogicalResult
printOperation(VitisCppEmitter &emitter, vitis::WhileOp whileOp)
{
    VitisCppEmitter::Scope scope(emitter);
    raw_indented_ostream &os = emitter.ostream();

    os << "while(" << emitter.getOrCreateName(whileOp.getCondition())
       << ") {\n#pragma HLS PIPELINE II=1\n";
    os.indent();

    Region::BlockListType &blocks = whileOp.getBody().getBlocks();
    for (Block &block : blocks) emitter.getOrCreateName(block);

    for (auto &opi : whileOp.getBody().getOps())
        if (failed(emitter.emitOperation(opi, false))) return failure();

    os.unindent() << "}";

    return success();
}

static LogicalResult
printOperation(VitisCppEmitter &emitter, vitis::WhileTrueOp whileTrueOp)
{
    VitisCppEmitter::Scope scope(emitter);
    raw_indented_ostream &os = emitter.ostream();

    os << "while(true) {\n#pragma HLS PIPELINE II=1\n";
    os.indent();

    Region::BlockListType &blocks = whileTrueOp.getBody().getBlocks();
    for (Block &block : blocks) emitter.getOrCreateName(block);

    for (auto &opi : whileTrueOp.getBody().getOps())
        if (failed(emitter.emitOperation(opi, false))) return failure();

    os.unindent() << "}";

    return success();
}

static LogicalResult
printOperation(VitisCppEmitter &emitter, vitis::IfBreakOp ifBreakOp)
{
    raw_indented_ostream &os = emitter.ostream();

    os << "if (" << emitter.getOrCreateName(ifBreakOp.getCondition())
       << ") break;";

    return success();
}

//===----------------------------------------------------------------------===//
// StreamOps
//===----------------------------------------------------------------------===//

static LogicalResult
printOperation(VitisCppEmitter &emitter, vitis::StreamReadOp readOp)
{
    raw_indented_ostream &os = emitter.ostream();

    if (failed(emitter.emitAssignPrefix(*readOp.getOperation())))
        return failure();

    os << emitter.getOrCreateName(readOp.getStream()) << ".read();";

    return success();
}

static LogicalResult
printOperation(VitisCppEmitter &emitter, vitis::StreamWriteOp writeOp)
{
    raw_indented_ostream &os = emitter.ostream();

    os << emitter.getOrCreateName(writeOp.getStream()) << ".write("
       << emitter.getOrCreateName(writeOp.getDataPkt()) << ");";

    return success();
}

static LogicalResult
printOperation(VitisCppEmitter &emitter, vitis::StreamGetLastOp getLastOp)
{
    raw_indented_ostream &os = emitter.ostream();

    if (failed(emitter.emitAssignPrefix(*getLastOp.getOperation())))
        return failure();

    os << emitter.getOrCreateName(getLastOp.getDataPkt()) << ".last;";

    return success();
}

static LogicalResult
printOperation(VitisCppEmitter &emitter, vitis::StreamSetLastOp setLastOp)
{
    raw_indented_ostream &os = emitter.ostream();

    os << emitter.getOrCreateName(setLastOp.getDataPkt())
       << ".last = " << emitter.getOrCreateName(setLastOp.getLast()) << ";";

    return success();
}

static LogicalResult
printOperation(VitisCppEmitter &emitter, vitis::StreamGetDataOp getDataOp)
{
    raw_indented_ostream &os = emitter.ostream();

    if (failed(emitter.emitAssignPrefix(*getDataOp.getOperation())))
        return failure();

    os << emitter.getOrCreateName(getDataOp.getDataPkt()) << ".data;";

    return success();
}

static LogicalResult
printOperation(VitisCppEmitter &emitter, vitis::StreamSetDataOp setDataOp)
{
    raw_indented_ostream &os = emitter.ostream();

    os << emitter.getOrCreateName(setDataOp.getDataPkt())
       << ".data = " << emitter.getOrCreateName(setDataOp.getData()) << ";";

    return success();
}

//===----------------------------------------------------------------------===//
// ArithOps
//===----------------------------------------------------------------------===//

static LogicalResult
printOperation(VitisCppEmitter &emitter, vitis::ArithAddOp addOp)
{
    raw_indented_ostream &os = emitter.ostream();

    if (failed(emitter.emitAssignPrefix(*addOp.getOperation())))
        return failure();

    os << emitter.getOrCreateName(addOp.getLhs()) << " + "
       << emitter.getOrCreateName(addOp.getRhs()) << ";";

    return success();
}

static LogicalResult
printOperation(VitisCppEmitter &emitter, vitis::ArithMulOp mulOp)
{
    raw_indented_ostream &os = emitter.ostream();

    if (failed(emitter.emitAssignPrefix(*mulOp.getOperation())))
        return failure();

    os << emitter.getOrCreateName(mulOp.getLhs()) << " * "
       << emitter.getOrCreateName(mulOp.getRhs()) << ";";

    return success();
}

static LogicalResult
printOperation(VitisCppEmitter &emitter, vitis::ArithOrOp orOp)
{
    raw_indented_ostream &os = emitter.ostream();

    if (failed(emitter.emitAssignPrefix(*orOp.getOperation())))
        return failure();

    os << emitter.getOrCreateName(orOp.getLhs()) << " | "
       << emitter.getOrCreateName(orOp.getRhs()) << ";";

    return success();
}

//===----------------------------------------------------------------------===//
// Emitter Functions
//===----------------------------------------------------------------------===//

LogicalResult
VitisCppEmitter::emitOperation(Operation &op, bool trailingSemicolon)
{
    LogicalResult status =
        llvm::TypeSwitch<Operation*, LogicalResult>(&op)
            .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
            .Case<vitis::FuncOp, vitis::ReturnOp>(
                [&](auto op) { return printOperation(*this, op); })
            .Case<
                vitis::ConstantOp,
                vitis::DefineOp,
                vitis::UpdateOp,
                vitis::WhileOp,
                vitis::WhileTrueOp,
                vitis::IfBreakOp>(
                [&](auto op) { return printOperation(*this, op); })
            .Case<
                vitis::StreamReadOp,
                vitis::StreamWriteOp,
                vitis::StreamGetLastOp,
                vitis::StreamSetLastOp,
                vitis::StreamGetDataOp,
                vitis::StreamSetDataOp>(
                [&](auto op) { return printOperation(*this, op); })
            .Case<vitis::ArithAddOp, vitis::ArithMulOp, vitis::ArithOrOp>(
                [&](auto op) { return printOperation(*this, op); })
            .Default(
                [](auto op) { return op->emitError("unsupported operation"); });

    if (failed(status)) return failure();
    os << (trailingSemicolon ? ";\n" : "\n");
    return success();
}

LogicalResult VitisCppEmitter::emitType(Location loc, Type vitisType)
{
    if (auto type = dyn_cast<IntegerType>(vitisType)) {
        auto datawidth = type.getWidth();
        if (type.getSignedness() == IntegerType::Unsigned)
            return (os << "ap_uint<" << datawidth << ">"), success();
        else
            return (os << "ap_int<" << datawidth << ">"), success();
    }
    if (auto type = dyn_cast<APAxiUType>(vitisType)) {
        auto last = type.getTlast() ? ", AXIS_ENABLE_LAST>" : ">";
        os << "ap_axiu<" << type.getDatawidth() << ", " << type.getKeep()
           << ", " << type.getUser() << ", " << type.getDest() << last;
        return success();
    }
    if (auto type = dyn_cast<APAxiSType>(vitisType)) {
        auto last = type.getTlast() ? ", AXIS_ENABLE_LAST>" : ">";
        os << "ap_axis<" << type.getDatawidth() << ", " << type.getKeep()
           << ", " << type.getUser() << ", " << type.getDest() << last;
        return success();
    }
    if (auto type = dyn_cast<StreamType>(vitisType)) {
        os << "hls::stream<";
        if (failed(emitType(loc, type.getStreamType()))) return failure();
        os << ">";
        return success();
    }

    return emitError(loc, "cannot emit type ") << vitisType;
}

LogicalResult VitisCppEmitter::emitTypes(Location loc, ArrayRef<Type> types)
{
    switch (types.size()) {
    case 0: os << "void"; return success();
    case 1: return emitType(loc, types.front());
    default: return emitTupleType(loc, types);
    }
}

LogicalResult VitisCppEmitter::emitTupleType(Location loc, ArrayRef<Type> types)
{
    os << "std::tuple<";
    if (failed(interleaveCommaWithError(types, os, [&](Type type) {
            return emitType(loc, type);
        })))
        return failure();
    os << ">";
    return success();
}

LogicalResult vitis::translateToVitisCpp(Operation* op, raw_ostream &os)
{
    VitisCppEmitter emitter(os);
    return emitter.emitOperation(*op, false);
}
