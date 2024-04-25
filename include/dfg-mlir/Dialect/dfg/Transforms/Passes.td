//===- Passes.td - Conversion passes ------------------------*- tablegen -*-===//
//
// This is the definitions file for the dfg-mlir dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_MLIR_TRANSFORM_PASSES
#define DFG_MLIR_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def DfgOperatorToProcess : Pass<"dfg-operator-to-process">
{
    let summary = "convert each operator to process";

    let constructor = "mlir::dfg::createDfgOperatorToProcessPass()";
}

#endif //DFG_MLIR_TRANSFORM_PASSES
