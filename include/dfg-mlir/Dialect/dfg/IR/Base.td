//===- Base.td ---------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the DFG dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_BASE
#define DFG_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def Dfg_Dialect : Dialect {
    let name = "dfg";
    let cppNamespace = "::mlir::dfg";

    // TODO: Draw the rest of the fucking owl.
}

#endif