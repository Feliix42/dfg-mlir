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

def LinalgInlineScalarArgument : Pass<"linalg-inline-scalar-argument">
{
    let summary = "inline scalar arguments in generic inputs";

    let constructor = "mlir::linalg::createLinalgInlineScalarArgumentPass()";
}

def MergingRegions : Pass<"dfg-merge-regions", "ModuleOp"> {
  let summary = "Merge multiple DFG regions into a single region";
  let description = [{
    Identifies and merges multiple DFG regions for MDC flow preparation.
  }];
  let dependentDialects = ["mlir::dfg::DfgDialect"];
  
  let constructor = "mlir::dfg::createMergingRegionsPass()";
}
//===----------------------------------------------------------------------===//
// InsertSBox Pass
//===----------------------------------------------------------------------===//

def InsertSBox : Pass<"insert-sbox", "mlir::ModuleOp"> {
  let summary = "Insert @sbox12int32 process if missing";
  let description = [{
    This pass inserts a fixed @sbox12int32 process into the IR if it is not already present.
    The inserted process contains a loop that pulls an input, adds 1, and pushes to two outputs.
  }];
  let constructor = "mlir::dfg::createInsertSBoxPass()";
  let dependentDialects = ["mlir::dfg::DfgDialect", "mlir::arith::ArithDialect"];
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

    let options = [
        Option<"printToPdf", "print-to-pdf", "bool", /*default=*/"0",
               "Print the graph to pdf files instead of dot.">
    ];
}

def DfgPrintOperatorToYaml : Pass<"print-operator-to-yaml">
{
    let summary = "print the inner graph within an operator to yaml file";

    let constructor = "mlir::dfg::createDfgPrintOperatorToYamlPass()";
}

def MemrefRemoveScalarGlobals : Pass<"memref-remove-scalar-globals">
{
    let summary = "remove the scalar memref globals from the graph";

    let constructor = "mlir::memref::createMemrefRemoveScalarGlobalsPass()";
}

def FuncRetrieveGraphLayers : Pass<"func-retrieve-graph-layers">
{
    let summary = "break the main func into different layer functions";

    let constructor = "mlir::func::createFuncRetrieveGraphLayersPass()";

    let options = [
        Option<"infoPath", "info-path", "std::string", /*default=*/"",
            "The path to the information json file">
    ];
}

def LinalgSoftTranspose : Pass<"linalg-soft-transpose">
{
    let summary = "transpose the tensors in the pass";

    let constructor = "mlir::linalg::createLinalgSoftTransposePass()";
}

#endif //DFG_MLIR_TRANSFORM_PASSES
