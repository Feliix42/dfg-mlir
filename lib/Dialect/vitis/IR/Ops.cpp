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
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#define DEBUG_TYPE "vitis-ops"

using namespace mlir;
using namespace mlir::vitis;

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
        if (!isFloat)
            initTy = IntegerType::get(parser.getContext(), bitwidth);
        else {
            switch (bitwidth) {
            case 16: initTy = FloatType::getF16(parser.getContext()); break;
            case 32: initTy = FloatType::getF32(parser.getContext()); break;
            case 64: initTy = FloatType::getF64(parser.getContext()); break;
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
    if (funcTy.getNumResults() != 0)
        return ::emitError(getLoc(), "Now only void return type is supported.");
    else if (funcTy.getNumInputs() != 0) {
        for (auto type : funcTy.getInputs())
            if (!isa<StreamType>(type))
                return ::emitError(
                    getLoc(),
                    "Now only stream or alias is supported as argument type.");
    }
    return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

// LogicalResult ReturnOp::verify()
// {
//     for (auto type : getOperandTypes())
//         if (!isa<IntegerType>(type))
//             return ::emitError(
//                 getLoc(),
//                 "Only support integers for vitis code generation.");
//     return success();
// }

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

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
// VitisDialect
//===----------------------------------------------------------------------===//

void VitisDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "dfg-mlir/Dialect/vitis/IR/Ops.cpp.inc"
        >();
}
