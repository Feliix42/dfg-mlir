/// Implements the vitis dialect ops.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Dialect/vitis/IR/Ops.h"

#include "dfg-mlir/Dialect/vitis/Enums.h"
#include "dfg-mlir/Dialect/vitis/IR/Dialect.h"
#include "dfg-mlir/Dialect/vitis/IR/Types.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <string>

#define DEBUG_TYPE "vitis-ops"

using namespace mlir;
using namespace mlir::vitis;

static Type getI1SameShape(Type type)
{
    auto i1Type = IntegerType::get(type.getContext(), 1);
    if (auto shapedType = llvm::dyn_cast<ShapedType>(type))
        return shapedType.cloneWith(std::nullopt, i1Type);
    return i1Type;
}

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "dfg-mlir/Dialect/vitis/IR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// IncludeOp
//===----------------------------------------------------------------------===//

ParseResult IncludeOp::parse(OpAsmParser &parser, OperationState &result)
{
    StringAttr include;
    OptionalParseResult includeParseResult =
        parser.parseOptionalAttribute(include, "include", result.attributes);
    if (!includeParseResult.has_value())
        return parser.emitError(parser.getNameLoc())
               << "expected string attribute";

    return success();
}

void IncludeOp::print(OpAsmPrinter &p) { p << " \"" << getInclude() << "\""; }

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

void VariableOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn)
{
    auto thisOp = getOperation();
    auto parentFunc = thisOp->getParentOfType<FuncOp>();
    if (!parentFunc) return;

    auto resultTy = getType();
    unsigned count = 0;
    parentFunc.walk([&](VariableOp varOp) -> WalkResult {
        if (varOp.getOperation() == thisOp) return WalkResult::interrupt();
        if ((varOp.getType() == resultTy)
            && (varOp.isVariableConst() == isVariableConst()))
            ++count;
        return WalkResult::advance();
    });

    std::string prefix = isVariableConst() ? "const" : "var";
    std::string suffix;
    if (isa<IntegerType>(resultTy))
        suffix = "_int_";
    else if (isa<IndexType>(resultTy))
        suffix = "_index_";
    else if (isa<FloatType>(resultTy))
        suffix = "_float_";
    else if (isa<ArrayType>(resultTy))
        suffix = "_array_";
    else if (isa<StreamType>(resultTy))
        suffix = "_stream_";
    setNameFn(getResult(), prefix + suffix + std::to_string(count));
}

void VariableOp::build(
    OpBuilder &builder,
    OperationState &result,
    Type type,
    Attribute init,
    bool is_const)
{
    result.addTypes(type);
    if (init) result.addAttribute(getInitAttrName(result.name), init);
    if (is_const) result.addAttribute("is_const", builder.getUnitAttr());
}

ParseResult VariableOp::parse(OpAsmParser &parser, OperationState &result)
{
    Type type;
    if (failed(parser.parseKeyword("as")))
        return parser.emitError(
            parser.getCurrentLocation(),
            "expected keyword 'as'");
    // If it's const value
    bool isConst = false;
    if (succeeded(parser.parseOptionalKeyword("const"))) {
        result.addAttribute("is_const", parser.getBuilder().getUnitAttr());
        isConst = true;
    }
    if (failed(parser.parseType(type)))
        return parser.emitError(parser.getCurrentLocation(), "expected type");
    result.addTypes(type);

    Attribute initAttr;
    if (succeeded(parser.parseOptionalEqual())) {
        if (failed(parser.parseAttribute(
                initAttr,
                type,
                getInitAttrName(result.name).data(),
                result.attributes)))
            return parser.emitError(
                parser.getCurrentLocation(),
                "expected initial attribute");
    } else {
        if (isConst)
            return parser.emitError(
                parser.getCurrentLocation(),
                "const value must have initial attribute");
    }

    return success();
}

void VariableOp::print(OpAsmPrinter &p)
{
    p << " as ";
    if (isVariableConst()) p << "const ";
    p << getType();
    if (getInit()) {
        p << " = ";
        p.printAttributeWithoutType(getInitAttr());
    }
}

LogicalResult VariableOp::verify()
{
    if (isa<PointerType>(getType()))
        return ::emitError(
            getLoc(),
            "Dynamic memory usage is not supported in HLS.");
    if (getInit()
        && !isa<IntegerType, IndexType, FloatType, ArrayType>(getType()))
        return ::emitError(getLoc(), "Unsupported type to have init value.");
    return success();
}

namespace {
struct RemoveUnusedVariable final : public OpRewritePattern<VariableOp> {
    using OpRewritePattern<VariableOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(VariableOp op, PatternRewriter &rewriter) const override
    {
        if (op.getResult().getUses().empty()) {
            rewriter.eraseOp(op);
            return success();
        }
        return failure();
    }
};

struct RemoveSameConstVariable final : public OpRewritePattern<VariableOp> {
    using OpRewritePattern<VariableOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(VariableOp op, PatternRewriter &rewriter) const override
    {
        // If there is not a const value, pass
        if (!op.isVariableConst()) return failure();

        auto initAttr = op.getInitAttr();
        auto constTy = op.getType();
        auto parentFunc = op.getOperation()->getParentOfType<FuncOp>();
        // Store all the variable op before current one
        SmallVector<VariableOp> constVarInFunc;
        parentFunc->walk([&](VariableOp varOp) -> WalkResult {
            if (varOp.isVariableConst()) {
                if (varOp != op) {
                    constVarInFunc.push_back(varOp);
                    return WalkResult::advance();
                } else {
                    return WalkResult::interrupt();
                }
            }
            return WalkResult::advance();
        });
        // Reverse to make sure dominance
        std::reverse(constVarInFunc.begin(), constVarInFunc.end());
        // Find the nearest same const, and replace with it
        for (auto varOp : constVarInFunc) {
            // TODO
            // If it's not define in the scope of this function, exit
            if (varOp->getParentOp() != parentFunc) break;
            // Else, check if it's the same const value
            if (varOp.getInitAttr() == initAttr && varOp.getType() == constTy) {
                op.getResult().replaceAllUsesWith(varOp.getResult());
                rewriter.eraseOp(op);
                return success();
            }
        }
        return failure();
    }
};

struct RemoveUnitDimension final : public OpRewritePattern<VariableOp> {
    using OpRewritePattern<VariableOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(VariableOp op, PatternRewriter &rewriter) const override
    {
        auto arrayType = dyn_cast<ArrayType>(op.getType());
        if (!arrayType) return failure();

        auto shape = arrayType.getShape();
        auto elementTy = arrayType.getElementType();

        // If any dimension is unit
        bool hasUnitDim =
            llvm::any_of(shape, [](int64_t dim) { return dim == 1; });
        if (!hasUnitDim) return failure();

        // Create new array
        SmallVector<int64_t> newShape;
        SmallVector<int64_t> collapsedDims;
        for (int64_t i = 0; i < (int64_t)shape.size(); ++i)
            if (shape[i] != 1)
                newShape.push_back(shape[i]);
            else
                collapsedDims.push_back(i);
        auto newArrayType = ArrayType::get(newShape, elementTy);
        auto newArrayVar =
            rewriter.create<VariableOp>(op.getLoc(), newArrayType);

        // If there is initial value
        if (op.getInit()) {
            auto initAttr = op.getInitAttr();
            if (auto denseAttr = dyn_cast<DenseElementsAttr>(initAttr)) {
                auto values = denseAttr.getValues<Attribute>();
                SmallVector<Attribute> valuesAttr(values.begin(), values.end());
                auto newDenseAttr =
                    DenseElementsAttr::get(newArrayType, valuesAttr);
                newArrayVar.setInitAttr(newDenseAttr);
                if (op.isVariableConst())
                    newArrayVar.setIsConstAttr(rewriter.getUnitAttr());
            }
        }
        // Add attributes to mark the original shape and the collapsed dimension
        // newArrayVar->setAttr(
        //     "original_shape",
        //     rewriter.getIndexArrayAttr(shape));
        newArrayVar->setAttr(
            "collapsed_dims",
            rewriter.getIndexArrayAttr(collapsedDims));

        op.getResult().replaceAllUsesWith(newArrayVar.getResult());
        rewriter.eraseOp(op);
        return success();
    }
};
} // namespace

void VariableOp::getCanonicalizationPatterns(
    RewritePatternSet &results,
    MLIRContext* context)
{
    results.add<
        RemoveUnusedVariable,
        RemoveSameConstVariable,
        RemoveUnitDimension>(context);
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result)
{
    auto buildFuncType = [](Builder &builder,
                            ArrayRef<Type> argTypes,
                            ArrayRef<Type> results,
                            function_interface_impl::VariadicFlag,
                            std::string &) {
        return builder.getFunctionType(argTypes, results);
    };

    return function_interface_impl::parseFunctionOp(
        parser,
        result,
        /*allowVariadic=*/false,
        getFunctionTypeAttrName(result.name),
        buildFuncType,
        getArgAttrsAttrName(result.name),
        getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter &p)
{
    function_interface_impl::printFunctionOp(
        p,
        *this,
        /*isVariadic=*/false,
        getFunctionTypeAttrName(),
        getArgAttrsAttrName(),
        getResAttrsAttrName());
}

LogicalResult FuncOp::verify()
{
    auto funcTy = getFunctionType();
    if (funcTy.getNumInputs() != 0) {
        for (auto type : funcTy.getInputs())
            if (!isa<StreamType, PointerType>(type))
                return ::emitError(
                    getLoc(),
                    "Now only stream or pointer type is supported as "
                    "argument "
                    "type.");
    }
    return success();
}

ArrayRef<Type> FuncOp::getFuncElementTypes()
{
    static SmallVector<Type> types;
    types.clear();

    for (auto type : getFunctionType().getInputs())
        if (auto streamType = dyn_cast<StreamType>(type))
            types.push_back(streamType.getStreamType());
        else if (auto ptrType = dyn_cast<PointerType>(type))
            types.push_back(ptrType.getPointerType());

    return types;
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

void ForOp::build(
    OpBuilder &builder,
    OperationState &result,
    int64_t lb,
    int64_t ub,
    int64_t step)
{
    OpBuilder::InsertionGuard g(builder);
    result.addAttribute(
        getLowerBoundAttrName(result.name),
        builder.getIntegerAttr(builder.getIndexType(), lb));
    result.addAttribute(
        getUpperBoundAttrName(result.name),
        builder.getIntegerAttr(builder.getIndexType(), ub));
    result.addAttribute(
        getStepAttrName(result.name),
        builder.getIntegerAttr(builder.getIndexType(), step));
    Region* bodyRegion = result.addRegion();
    Block* bodyBlock = builder.createBlock(bodyRegion);
    bodyBlock->addArgument(builder.getIndexType(), result.location);
}

void ForOp::build(
    OpBuilder &builder,
    OperationState &result,
    Attribute lb,
    Attribute ub,
    Attribute step)
{
    OpBuilder::InsertionGuard g(builder);
    result.addAttribute(getLowerBoundAttrName(result.name), lb);
    result.addAttribute(getUpperBoundAttrName(result.name), ub);
    result.addAttribute(getStepAttrName(result.name), step);
    Region* bodyRegion = result.addRegion();
    Block* bodyBlock = builder.createBlock(bodyRegion);
    bodyBlock->addArgument(builder.getIndexType(), result.location);
}

void ForOp::getAsmBlockArgumentNames(
    Region &region,
    OpAsmSetValueNameFn setNameFn)
{
    Operation* forOp = getOperation();
    unsigned nestedLevel = 0;

    while ((forOp = forOp->getParentOp()) && forOp)
        if (isa<ForOp>(forOp)) ++nestedLevel;

    setNameFn(region.getArgument(0), "idx" + std::to_string(nestedLevel));
}

ParseResult ForOp::parse(OpAsmParser &parser, OperationState &result)
{
    auto &builder = parser.getBuilder();
    Type type = builder.getIndexType();

    OpAsmParser::Argument inductionVariable;
    if (parser.parseOperand(inductionVariable.ssaName) || parser.parseEqual())
        return failure();

    IntegerAttr lbAttr, ubAttr, stepAttr;
    if (parser.parseAttribute(
            lbAttr,
            builder.getIndexType(),
            getLowerBoundAttrName(result.name).data(),
            result.attributes)
        || parser.parseKeyword("to")
        || parser.parseAttribute(
            ubAttr,
            builder.getIndexType(),
            getUpperBoundAttrName(result.name).data(),
            result.attributes)
        || parser.parseKeyword("step")
        || parser.parseAttribute(
            stepAttr,
            builder.getIndexType(),
            getStepAttrName(result.name).data(),
            result.attributes))
        return failure();

    SmallVector<OpAsmParser::Argument> args;
    args.push_back(inductionVariable);
    args.front().type = type;

    Region* body = result.addRegion();
    if (parser.parseRegion(*body, args)) return failure();

    if (parser.parseOptionalAttrDict(result.attributes)) return failure();

    return success();
}

void ForOp::print(OpAsmPrinter &p)
{
    p << " " << getInductionVar() << " = " << getLowerBound() << " to "
      << getUpperBound() << " step " << getStep();
    p << ' ';
    p.printRegion(
        getRegion(),
        /*printEntryBlockArgs=*/false,
        /*printBlockTerminators=*/false);
}

LogicalResult ForOp::verify()
{
    if (getLowerBound() == getUpperBound())
        return ::emitError(getLoc(), "expected loop iterate at least once");
    if (getStep() == 0) return ::emitError(getLoc(), "expected non-zero step");
    if (getUpperBound().slt(getLowerBound() + getStep()))
        return ::emitError(getLoc(), "unexpected index overflow");
    if (getRegion().getOps().empty())
        return ::emitError(getLoc(), "expected non-empty loop body");
    return success();
}

namespace {
struct RemoveTrivialLoop : public OpRewritePattern<ForOp> {
    using OpRewritePattern<ForOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ForOp op, PatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        rewriter.setInsertionPoint(op);
        // If the loop has only one iteration, inline the content of this loop
        // and remove it
        if ((op.getLowerBound() + op.getStep()) == op.getUpperBound()) {
            // Create a constant to replace the induction variable
            IRMapping mapper;
            auto constVarOp = rewriter.create<VariableOp>(
                loc,
                rewriter.getIndexType(),
                op.getLowerBoundAttr(),
                true);
            mapper.map(op.getInductionVar(), constVarOp.getResult());
            // Clone the contents at the current location
            for (auto &opi : op.getRegion().getOps())
                rewriter.clone(opi, mapper);
            rewriter.eraseOp(op);
        }
        return success();
    }
};
} // namespace

void ForOp::getCanonicalizationPatterns(
    RewritePatternSet &results,
    MLIRContext* context)
{
    results.add<RemoveTrivialLoop>(context);
}

//===----------------------------------------------------------------------===//
// ArithOps
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ArithSelectOp
//===----------------------------------------------------------------------===//

ParseResult ArithSelectOp::parse(OpAsmParser &parser, OperationState &result)
{
    Type conditionType, resultType;
    SmallVector<OpAsmParser::UnresolvedOperand, 3> operands;
    if (parser.parseOperandList(operands, /*requiredOperandCount=*/3)
        || parser.parseOptionalAttrDict(result.attributes)
        || parser.parseColonType(resultType))
        return failure();

    conditionType = parser.getBuilder().getI1Type();
    result.addTypes(resultType);
    if (parser.resolveOperands(
            operands,
            {conditionType, resultType, resultType},
            parser.getNameLoc(),
            result.operands))
        return failure();

    return success();
}

void ArithSelectOp::print(OpAsmPrinter &p)
{
    p << " " << getOperands();
    p.printOptionalAttrDict((*this)->getAttrs());
    p << " : ";
    p << getType();
}

//===----------------------------------------------------------------------===//
// ArrayOps
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ArrayReadOp
//===----------------------------------------------------------------------===//

LogicalResult ArrayReadOp::verify()
{
    if (static_cast<int64_t>(getIndices().size()) != getArrayType().getRank()) {
        return emitOpError(
                   "incorrect number of indices for array read, expected ")
               << getArrayType().getRank() << " but got "
               << getIndices().size();
    }
    return success();
}

namespace {
struct AdjustArrayReadForRemovedDims final
        : public OpRewritePattern<ArrayReadOp> {
    using OpRewritePattern<ArrayReadOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ArrayReadOp op, PatternRewriter &rewriter) const override
    {
        // If already adjusted
        if (op->hasAttr("adjusted")) return failure();

        Value array = op.getArray();
        auto definingOp = array.getDefiningOp();

        // If the array's dimension is not removed
        if (!definingOp || !definingOp->hasAttr("collapsed_dims"))
            return failure();

        // Get removed dimensions
        auto collapsedDimsAttr =
            cast<ArrayAttr>(definingOp->getAttr("collapsed_dims"));
        SmallVector<int64_t> collapsedDims;
        for (auto attr : collapsedDimsAttr)
            collapsedDims.push_back(cast<IntegerAttr>(attr).getInt());

        // Get new indices
        SmallVector<Value> newIndices;
        for (size_t i = 1; i < op.getNumOperands(); ++i) {
            // If the index' dim is not removed, keep it
            if (!llvm::is_contained(collapsedDims, i - 1))
                newIndices.push_back(op.getOperand(i));
        }

        auto newRead = rewriter.create<ArrayReadOp>(
            op.getLoc(),
            op.getResult().getType(),
            array,
            newIndices);
        newRead->setAttr("adjusted", rewriter.getUnitAttr());

        rewriter.replaceOp(op, newRead.getResult());
        return success();
    }
};
} // namespace

void ArrayReadOp::getCanonicalizationPatterns(
    RewritePatternSet &results,
    MLIRContext* context)
{
    results.add<AdjustArrayReadForRemovedDims>(context);
}

//===----------------------------------------------------------------------===//
// ArrayWriteOp
//===----------------------------------------------------------------------===//

LogicalResult ArrayWriteOp::verify()
{
    if (getNumOperands() != 2 + getArrayType().getRank())
        return emitOpError(
            "array write index operand count not equal to memref rank");
    return success();
}

namespace {
struct AdjustArrayWriteForRemovedDims final
        : public OpRewritePattern<ArrayWriteOp> {
    using OpRewritePattern<ArrayWriteOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ArrayWriteOp op, PatternRewriter &rewriter) const override
    {
        // If already adjusted
        if (op->hasAttr("adjusted")) return failure();

        Value array = op.getArray();
        auto definingOp = array.getDefiningOp();

        // If the array's dimension is not removed
        if (!definingOp || !definingOp->hasAttr("collapsed_dims"))
            return failure();

        // Get removed dimensions
        auto collapsedDimsAttr =
            cast<ArrayAttr>(definingOp->getAttr("collapsed_dims"));
        SmallVector<int64_t> collapsedDims;
        for (auto attr : collapsedDimsAttr)
            collapsedDims.push_back(cast<IntegerAttr>(attr).getInt());

        // Create new indices
        SmallVector<Value> newIndices;
        for (size_t i = 2; i < op.getNumOperands(); ++i) {
            // If the index' dim is not removed, keep it
            if (!llvm::is_contained(collapsedDims, i - 2))
                newIndices.push_back(op.getOperand(i));
        }

        auto newWrite = rewriter.create<ArrayWriteOp>(
            op.getLoc(),
            op.getValue(),
            array,
            newIndices);
        newWrite->setAttr("adjusted", rewriter.getUnitAttr());

        rewriter.replaceOp(op, newWrite);
        return success();
    }
};
} // namespace

void ArrayWriteOp::getCanonicalizationPatterns(
    RewritePatternSet &results,
    MLIRContext* context)
{
    results.add<AdjustArrayWriteForRemovedDims>(context);
}

//===----------------------------------------------------------------------===//
// PragmaOps
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PragmaBindStorageOp
//===----------------------------------------------------------------------===//

void PragmaBindStorageOp::build(
    OpBuilder &builder,
    OperationState &result,
    Value variable,
    vitis::PragmaStorageType type,
    vitis::PragmaStorageImpl impl)
{
    result.addOperands(variable);
    result.addAttribute(
        getTypeAttrName(result.name),
        vitis::PragmaStorageTypeAttr::get(builder.getContext(), type));
    result.addAttribute(
        getImplAttrName(result.name),
        vitis::PragmaStorageImplAttr::get(builder.getContext(), impl));
}

ParseResult
PragmaBindStorageOp::parse(OpAsmParser &parser, OperationState &result)
{
    OpAsmParser::UnresolvedOperand variableOperand;
    Type variableType;
    if (parser.parseKeyword("variable") || parser.parseEqual()
        || parser.parseOperand(variableOperand) || parser.parseLParen()
        || parser.parseType(variableType) || parser.parseRParen())
        return failure();

    StringRef typeStr;
    if (parser.parseKeyword("type") || parser.parseEqual()) return failure();
    if (parser.parseKeyword(&typeStr)) return failure();

    vitis::PragmaStorageType storageType;
    if (typeStr == "fifo")
        storageType = vitis::PragmaStorageType::fifo;
    else if (typeStr == "ram_1p")
        storageType = vitis::PragmaStorageType::ram_1p;
    else if (typeStr == "ram_1wnr")
        storageType = vitis::PragmaStorageType::ram_1wnr;
    else if (typeStr == "ram_2p")
        storageType = vitis::PragmaStorageType::ram_2p;
    else if (typeStr == "ram_s2p")
        storageType = vitis::PragmaStorageType::ram_s2p;
    else if (typeStr == "ram_t2p")
        storageType = vitis::PragmaStorageType::ram_t2p;
    else if (typeStr == "rom_1p")
        storageType = vitis::PragmaStorageType::rom_1p;
    else if (typeStr == "rom_2p")
        storageType = vitis::PragmaStorageType::rom_2p;
    else if (typeStr == "rom_np")
        storageType = vitis::PragmaStorageType::rom_np;
    else
        return parser.emitError(parser.getNameLoc(), "unknown storage type: ")
               << typeStr;

    StringRef implStr;
    if (parser.parseKeyword("impl") || parser.parseEqual()) return failure();
    if (parser.parseKeyword(&implStr)) return failure();

    vitis::PragmaStorageImpl storageImpl;
    if (implStr == "bram")
        storageImpl = vitis::PragmaStorageImpl::bram;
    else if (implStr == "bram_ecc")
        storageImpl = vitis::PragmaStorageImpl::bram_ecc;
    else if (implStr == "lutram")
        storageImpl = vitis::PragmaStorageImpl::lutram;
    else if (implStr == "uram")
        storageImpl = vitis::PragmaStorageImpl::uram;
    else if (implStr == "uram_ecc")
        storageImpl = vitis::PragmaStorageImpl::uram_ecc;
    else if (implStr == "srl")
        storageImpl = vitis::PragmaStorageImpl::srl;
    else if (implStr == "memory")
        storageImpl = vitis::PragmaStorageImpl::memory;
    else if (implStr == "auto")
        storageImpl = vitis::PragmaStorageImpl::automatic;
    else
        return parser.emitError(
                   parser.getNameLoc(),
                   "unknown storage implementation: ")
               << implStr;

    if (parser.resolveOperand(variableOperand, variableType, result.operands))
        return failure();
    result.addAttribute(
        getTypeAttrName(result.name).data(),
        vitis::PragmaStorageTypeAttr::get(parser.getContext(), storageType));
    result.addAttribute(
        getImplAttrName(result.name).data(),
        vitis::PragmaStorageImplAttr::get(parser.getContext(), storageImpl));

    return success();
}

void PragmaBindStorageOp::print(OpAsmPrinter &p)
{
    p << " variable=" << getVariable();
    p << "(" << getVariableType() << ")";
    p << " type=" << getType();
    p << " impl=";
    auto impl = getImpl();
    if (impl == vitis::PragmaStorageImpl::automatic)
        p << "auto";
    else
        p << impl;
}

//===----------------------------------------------------------------------===//
// PragmaDataflowOp
//===----------------------------------------------------------------------===//

void PragmaDataflowOp::build(
    OpBuilder &builder,
    OperationState &state,
    llvm::function_ref<void()> dataflowRegionCtor)
{
    OpBuilder::InsertionGuard guard(builder);
    Region* dataflowRegion = state.addRegion();
    if (dataflowRegionCtor) {
        builder.createBlock(dataflowRegion);
        dataflowRegionCtor();
    }
}

//===----------------------------------------------------------------------===//
// PragmaInlineOp
//===----------------------------------------------------------------------===//

void PragmaInlineOp::build(OpBuilder &builder, OperationState &result, bool off)
{
    if (off) result.addAttribute("off", builder.getUnitAttr());
}

ParseResult PragmaInlineOp::parse(OpAsmParser &parser, OperationState &result)
{
    if (succeeded(parser.parseOptionalKeyword("off")))
        result.addAttribute("off", parser.getBuilder().getUnitAttr());
    return success();
}

void PragmaInlineOp::print(OpAsmPrinter &p)
{
    if (getOff()) p << " off";
}

namespace {
struct HoistPragmaInlinePattern : public OpRewritePattern<PragmaInlineOp> {
    using OpRewritePattern<PragmaInlineOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(PragmaInlineOp op, PatternRewriter &rewriter) const override
    {
        Block* block = op->getBlock();
        if (&block->front() == op.getOperation()) return failure();
        op->moveBefore(&block->front());
        return success();
    }
    // Ensure this has higher piority over constant hoist
    PatternBenefit getBenefit() const { return 2; }
};
} // namespace
void PragmaInlineOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns,
    MLIRContext* context)
{
    patterns.add<HoistPragmaInlinePattern>(context, 100);
}

//===----------------------------------------------------------------------===//
// PragmaInterfaceOp
//===----------------------------------------------------------------------===//

void PragmaInterfaceOp::build(
    OpBuilder &builder,
    OperationState &result,
    PragmaInterfaceMode mode,
    Value port,
    StringRef bundle,
    std::optional<PragmaInterfaceMasterAxiOffset> offset)
{
    result.addAttribute(
        getModeAttrName(result.name),
        PragmaInterfaceModeAttr::get(builder.getContext(), mode));
    result.addOperands(port);
    result.addAttribute(
        getBundleAttrName(result.name),
        StringAttr::get(builder.getContext(), bundle));
    if (offset.has_value())
        result.addAttribute(
            getOffsetAttrName(result.name),
            PragmaInterfaceMasterAxiOffsetAttr::get(
                builder.getContext(),
                offset.value()));
}

ParseResult
PragmaInterfaceOp::parse(OpAsmParser &parser, OperationState &result)
{
    StringRef modeStr;
    if (parser.parseKeyword("mode") || parser.parseEqual()
        || parser.parseKeyword(&modeStr))
        return failure();

    PragmaInterfaceMode interfaceMode;
    if (modeStr == "m_axi")
        interfaceMode = PragmaInterfaceMode::m_axi;
    else if (modeStr == "s_axilite")
        interfaceMode = PragmaInterfaceMode::s_axilite;
    else
        return parser.emitError(parser.getNameLoc(), "unknown interface mode: ")
               << modeStr;

    OpAsmParser::UnresolvedOperand portOperand;
    Type portType;
    if (parser.parseKeyword("port") || parser.parseEqual()
        || parser.parseOperand(portOperand) || parser.parseLParen()
        || parser.parseType(portType) || parser.parseRParen())
        return failure();

    if (succeeded(parser.parseOptionalKeyword("offset"))) {
        if (parser.parseEqual()) return failure();
        StringRef offsetStr;
        if (parser.parseKeyword(&offsetStr)) return failure();

        vitis::PragmaInterfaceMasterAxiOffset axiOffset;
        if (offsetStr == "off")
            axiOffset = vitis::PragmaInterfaceMasterAxiOffset::off;
        else if (offsetStr == "direct")
            axiOffset = vitis::PragmaInterfaceMasterAxiOffset::direct;
        else if (offsetStr == "slave")
            axiOffset = vitis::PragmaInterfaceMasterAxiOffset::slave;
        else
            return parser.emitError(parser.getNameLoc(), "unknown axi offset: ")
                   << offsetStr;

        result.addAttribute(
            getOffsetAttrName(result.name).data(),
            PragmaInterfaceMasterAxiOffsetAttr::get(
                parser.getContext(),
                axiOffset));
    }

    StringRef bundleStr;
    if (parser.parseKeyword("bundle") || parser.parseEqual()
        || parser.parseKeyword(&bundleStr))
        return failure();

    if (parser.resolveOperand(portOperand, portType, result.operands))
        return failure();

    result.addAttribute(
        getModeAttrName(result.name).data(),
        PragmaInterfaceModeAttr::get(parser.getContext(), interfaceMode));
    result.addAttribute(
        getBundleAttrName(result.name).data(),
        StringAttr::get(parser.getContext(), bundleStr));

    return success();
}

void PragmaInterfaceOp::print(OpAsmPrinter &p)
{
    p << " mode=" << getMode();
    p << " port=" << getPort();
    p << "(" << getPortType() << ")";
    if (getOffset()) p << " offset=" << getOffset();
    p << " bundle=" << getBundle();
}

LogicalResult PragmaInterfaceOp::verify()
{
    if (getOffset() && getMode() != PragmaInterfaceMode::m_axi)
        return ::emitError(getLoc(), "expected m_axi mode for offset usage.");
    return success();
}

//===----------------------------------------------------------------------===//
// PragmaPipelineOp
//===----------------------------------------------------------------------===//

void PragmaPipelineOp::build(
    OpBuilder &builder,
    OperationState &result,
    int64_t interval,
    vitis::PragmaPipelineStyle style)
{
    result.addAttribute(
        getIntervalAttrName(result.name),
        builder.getI64IntegerAttr(interval));
    result.addAttribute(
        getStyleAttrName(result.name),
        vitis::PragmaPipelineStyleAttr::get(builder.getContext(), style));
}

ParseResult PragmaPipelineOp::parse(OpAsmParser &parser, OperationState &result)
{
    int64_t interval;
    if (parser.parseKeyword("II") || parser.parseEqual()) return failure();
    if (parser.parseInteger(interval)) return failure();

    StringRef styleStr;
    if (parser.parseKeyword("style") || parser.parseEqual()) return failure();
    if (parser.parseKeyword(&styleStr)) return failure();

    vitis::PragmaPipelineStyle style;
    if (styleStr == "flp")
        style = vitis::PragmaPipelineStyle::flp;
    else if (styleStr == "stp")
        style = vitis::PragmaPipelineStyle::stp;
    else if (styleStr == "frp")
        style = vitis::PragmaPipelineStyle::frp;
    else
        return parser.emitError(parser.getNameLoc(), "unknown pipeline style: ")
               << styleStr;

    result.addAttribute(
        getIntervalAttrName(result.name).data(),
        parser.getBuilder().getI64IntegerAttr(interval));
    result.addAttribute(
        "style",
        vitis::PragmaPipelineStyleAttr::get(parser.getContext(), style));

    return success();
}

void PragmaPipelineOp::print(OpAsmPrinter &p)
{
    p << " II=" << getInterval();
    p << " style=" << getStyle();
}

LogicalResult PragmaPipelineOp::verify() { return success(getInterval() > 0); }

//===----------------------------------------------------------------------===//
// PragmaStreamOp
//===----------------------------------------------------------------------===//

void PragmaStreamOp::build(
    OpBuilder &builder,
    OperationState &result,
    Value variable,
    int64_t depth)
{
    result.addOperands(variable);
    result.addAttribute(
        getDepthAttrName(result.name),
        builder.getI64IntegerAttr(depth));
}

ParseResult PragmaStreamOp::parse(OpAsmParser &parser, OperationState &result)
{
    OpAsmParser::UnresolvedOperand variableOperand;
    Type variableType;
    if (parser.parseKeyword("variable") || parser.parseEqual()
        || parser.parseOperand(variableOperand) || parser.parseLParen()
        || parser.parseType(variableType) || parser.parseRParen())
        return failure();

    int64_t depth;
    if (parser.parseKeyword("depth") || parser.parseEqual()) return failure();
    if (parser.parseInteger(depth)) return failure();

    if (parser.resolveOperand(variableOperand, variableType, result.operands))
        return failure();
    result.addAttribute(
        getDepthAttrName(result.name).data(),
        parser.getBuilder().getI64IntegerAttr(depth));

    return success();
}

void PragmaStreamOp::print(OpAsmPrinter &p)
{
    p << " variable=" << getVariable();
    p << "(" << getVariableType() << ")";
    p << " depth=" << getDepth();
}

LogicalResult PragmaStreamOp::verify() { return success(getDepth() > 0); }

//===----------------------------------------------------------------------===//
// VitisDialect
//===----------------------------------------------------------------------===//

void VitisDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "dfg-mlir/Dialect/vitis/IR/Ops.cpp.inc"
        >();
}
