//===- Passes.td - Conversion passes ------------------------*- tablegen -*-===//
//
// This is the definitions file for the dfg-mlir dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef VITIS_MLIR_TRANSFORM_PASSES
#define VITIS_MLIR_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def VitisInsertIncludes : Pass<"vitis-insert-includes">
{
    let summary = "insert include headers based on the content";

    let constructor = "mlir::vitis::createVitisInsertIncludesPass()";
}

def VitisMergeCastChain : Pass<"merge-cast-chain">
{
    let summary = "merge the chain of cast operations into one cast";

    let constructor = "mlir::vitis::createVitisMergeCastChainPass()";
}

#endif //VITIS_MLIR_TRANSFORM_PASSES
