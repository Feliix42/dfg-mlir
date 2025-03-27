/// Implements the vitis dialect ops.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Dialect/vitis/IR/Ops.h"

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

#include <functional>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

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
// VariableOp
//===----------------------------------------------------------------------===//

unsigned getBitwidth(Type type, bool &isFloat)
{
    if (auto intTy = dyn_cast<IntegerType>(type)) return intTy.getWidth();
    if (auto floatTy = dyn_cast<FloatType>(type)) {
        isFloat = true;
        return floatTy.getWidth();
    }
    if (auto fixedTy = dyn_cast<APFixedType>(type))
        return fixedTy.getDatawidth();
    if (auto fixedUTy = dyn_cast<APFixedUType>(type))
        return fixedUTy.getDatawidth();
    if (auto arrayTy = dyn_cast<vitis::ArrayType>(type))
        return getBitwidth(arrayTy.getElemType(), isFloat);
    return 0;
}

ParseResult VariableOp::parse(OpAsmParser &parser, OperationState &result)
{
    // Define variables for parsing
    OpAsmParser::UnresolvedOperand initOperand;
    Type variableType;

    // Attempt to parse the optional 'init' keyword and operand
    bool hasInit = succeeded(parser.parseOptionalKeyword("init"));
    if (hasInit) {
        if (parser.parseOperand(initOperand)) return failure();
    }

    // Parse attributes and result type
    if (parser.parseOptionalAttrDict(result.attributes)
        || parser.parseColonType(variableType))
        return failure();

    // Add the result type to the operation state
    result.addTypes(variableType);

    // Resolve the operand if it exists
    bool isFloat = false;
    unsigned bitwidth = getBitwidth(variableType, isFloat);
    if (hasInit) {
        Type initTy;
        if (!isFloat) {
            if (isa<IndexType>(variableType))
                initTy = variableType;
            else
                initTy = IntegerType::get(parser.getContext(), bitwidth);

        } else {
            switch (bitwidth) {
            case 16: initTy = Float16Type::get(parser.getContext()); break;
            case 32: initTy = Float32Type::get(parser.getContext()); break;
            case 64: initTy = Float64Type::get(parser.getContext()); break;
            default:
                return parser.emitError(
                    parser.getNameLoc(),
                    "Unsupported floating point bitwidth for init value.");
            }
        }

        if (parser.resolveOperand(initOperand, initTy, result.operands))
            return failure();
    }

    return success();
}

void VariableOp::print(OpAsmPrinter &p)
{
    // Print the optional 'init' operand
    if (getInit()) p << " init " << getInit();

    // Print attributes and result type
    p.printOptionalAttrDict((*this)->getAttrs());
    p << " : " << getType();
}

LogicalResult VariableOp::verify()
{
    if (getInit()) {
        if (isa<IndexType>(getInit().getType())) return success();
        auto initBitwidth = getInit().getType().getIntOrFloatBitWidth();
        auto type = getType();
        bool isFloat = false;
        unsigned bitwidth = getBitwidth(type, isFloat);
        if (initBitwidth != bitwidth)
            return ::emitError(
                getLoc(),
                "Different bitwidth between variable and the init value.");
    }
    return success();
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
                    "Now only stream or pointer type is supported as argument "
                    "type.");
    }
    return success();
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

void ForOp::build(
    OpBuilder &builder,
    OperationState &result,
    Value lb,
    Value ub,
    Value step)
{
    OpBuilder::InsertionGuard g(builder);
    result.addOperands({lb, ub, step});
    Type t = lb.getType();
    Region* bodyRegion = result.addRegion();
    Block* bodyBlock = builder.createBlock(bodyRegion);
    bodyBlock->addArgument(t, result.location);
}

ParseResult ForOp::parse(OpAsmParser &parser, OperationState &result)
{
    auto &builder = parser.getBuilder();
    Type type = builder.getIndexType();

    OpAsmParser::Argument inductionVariable;
    OpAsmParser::UnresolvedOperand lb, ub, step;

    if (parser.parseOperand(inductionVariable.ssaName) || parser.parseEqual()
        || parser.parseOperand(lb) || parser.parseKeyword("to")
        || parser.parseOperand(ub) || parser.parseKeyword("step")
        || parser.parseOperand(step))
        return failure();

    SmallVector<OpAsmParser::Argument> args;
    SmallVector<OpAsmParser::UnresolvedOperand> operands;
    args.push_back(inductionVariable);
    args.front().type = type;
    if (parser.resolveOperand(lb, type, result.operands)
        || parser.resolveOperand(ub, type, result.operands)
        || parser.resolveOperand(step, type, result.operands))
        return failure();

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
    p.printOptionalAttrDict((*this)->getAttrs());
}

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
// VitisDialect
//===----------------------------------------------------------------------===//

void VitisDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "dfg-mlir/Dialect/vitis/IR/Ops.cpp.inc"
        >();
}
