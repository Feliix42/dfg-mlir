/// Implements the dfg dialect ops.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#define DEBUG_TYPE "dfg-ops"

using namespace mlir;
using namespace mlir::dfg;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "dfg-mlir/Dialect/dfg/IR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// OperatorOp
//===----------------------------------------------------------------------===//

void OperatorOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   ArrayRef<NamedAttribute> inp_attrs,
                   ArrayRef<NamedAttribute> out_attrs) {
    state.addAttribute(SymbolTable::getSymbolAttrName(),
                       builder.getStringAttr(name));
    state.attributes.append(inp_attrs.begin(), inp_attrs.end());
    state.attributes.append(out_attrs.begin(), out_attrs.end());
    state.addRegion();
}

// temporary workaround
bool OperatorOp::isExternal() {
    Region &body = getRegion();
    return body.empty();
}

// TODO(feliix42): Change the parser to actually parse the format that I
//                 envision for the operator semantics
ParseResult OperatorOp::parse(OpAsmParser &parser, OperationState &result) {
//   auto buildFuncType =
//       [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
//          function_interface_impl::VariadicFlag,
//          std::string &) { return builder.getFunctionType(argTypes, results); };

//   return function_interface_impl::parseFunctionOp(
//       parser, result, /*allowVariadic=*/false,
//       getFunctionTypeAttrName(result.name), buildFuncType,
//       getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
    return ParseResult::failure();
}

void OperatorOp::print(OpAsmPrinter &p) {
//   function_interface_impl::printFunctionOp(
//       p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
//       getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// DfgDialect
//===----------------------------------------------------------------------===//

void DfgDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "dfg-mlir/Dialect/dfg/IR/Ops.cpp.inc"
        >();
}
