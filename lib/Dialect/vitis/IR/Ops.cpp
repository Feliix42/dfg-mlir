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

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
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

    unsigned bitwidth = 0;
    if (isa<IntegerType>(variableType))
        bitwidth = variableType.getIntOrFloatBitWidth();
    else {
        if (auto fixedTy = dyn_cast<APFixedType>(variableType))
            bitwidth = fixedTy.getDatawidth();
        if (auto fixedUTy = dyn_cast<APFixedUType>(variableType))
            bitwidth = fixedUTy.getDatawidth();
    }

    // Resolve the operand if it exists
    if (hasInit) {
        auto initTy = IntegerType::get(parser.getContext(), bitwidth);
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
        unsigned bitwidth = 0;
        if (isa<IntegerType>(type) || isa<FloatType>(type))
            bitwidth = type.getIntOrFloatBitWidth();
        else {
            if (auto fixedTy = dyn_cast<APFixedType>(type))
                bitwidth = fixedTy.getDatawidth();
            if (auto fixedUTy = dyn_cast<APFixedUType>(type))
                bitwidth = fixedUTy.getDatawidth();
        }

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
// VitisDialect
//===----------------------------------------------------------------------===//

void VitisDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "dfg-mlir/Dialect/vitis/IR/Ops.cpp.inc"
        >();
}
