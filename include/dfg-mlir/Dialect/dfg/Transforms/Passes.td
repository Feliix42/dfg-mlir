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

def DfgPrintOperatorToYaml : Pass<"print-operator-to-yaml">
{
    let summary = "print the inner graph within an operator to yaml file";

    let constructor = "mlir::dfg::createDfgPrintOperatorToYamlPass()";
}

#endif //DFG_MLIR_TRANSFORM_PASSES
