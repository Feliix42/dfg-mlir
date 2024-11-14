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

#include <circt/Support/LLVM.h>
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
// ConstantOp
//===----------------------------------------------------------------------===//

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result)
{
    IntegerAttr valueAttr;

    if (parser.parseAttribute(valueAttr, "value", result.attributes)
        || parser.parseOptionalAttrDict(result.attributes))
        return failure();

    result.addTypes(valueAttr.getType());
    return success();
}

void ConstantOp::print(OpAsmPrinter &p)
{
    p << " ";
    p.printAttribute(getValueAttr());
    p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
}

LogicalResult ConstantOp::verify()
{
    auto type = getType();
    if (!isa<IntegerType>(type))
        return ::emitError(
            getLoc(),
            "Only support integers for vitis code generation.");
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
            if (!isa<StreamType>(type) && !isa<AliasType>(type))
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
