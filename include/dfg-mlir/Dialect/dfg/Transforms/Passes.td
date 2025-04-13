//===- Passes.td - Conversion passes ------------------------*- tablegen -*-===//
//
// This is the definitions file for the dfg-mlir dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_MLIR_TRANSFORM_PASSES
#define DFG_MLIR_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def DfgInlineRegion : Pass<"dfg-inline-region">
{
    let summary = "inline region based on inline strategy";

    let constructor = "mlir::dfg::createDfgInlineRegionPass()";

    let options = [
        Option<"strategy", "strategy", "std::string", /*default=*/"\"full\"",
            "Strategy to inline regions">
    ];
}

def DfgLowerInsideToLinalg : Pass<"dfg-lower-inside-to-linalg">
{
    let summary = "lower the contents of operator/process to linalg level";

    let constructor = "mlir::dfg::createDfgLowerInsideToLinalgPass()";
}

def DfgOperatorToProcess : Pass<"dfg-operator-to-process">
{
    let summary = "convert each operator to process";

    let constructor = "mlir::dfg::createDfgOperatorToProcessPass()";
}

def DfgPrintGraph : Pass<"dfg-print-graph">
{
    let summary = "print the graph from dfg-mlir";

    let constructor = "mlir::dfg::createDfgPrintGraphPass()";
}

def DfgPrintOperatorToYaml : Pass<"print-operator-to-yaml">
{
    let summary = "print the inner graph within an operator to yaml file";

    let constructor = "mlir::dfg::createDfgPrintOperatorToYamlPass()";
}

#endif //DFG_MLIR_TRANSFORM_PASSES
