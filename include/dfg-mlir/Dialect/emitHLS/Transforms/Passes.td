//===- Passes.td - Conversion passes ------------------------*- tablegen -*-===//
//
// This is the definitions file for the dfg-mlir dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef EMITHLS_MLIR_TRANSFORM_PASSES
#define EMITHLS_MLIR_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def emitHLSInsertIncludes : Pass<"emitHLS-insert-includes">
{
    let summary = "insert include headers based on the content";

    let constructor = "mlir::emitHLS::createEmitHLSInsertIncludesPass()";
}

def emitHLSMergeCastChain : Pass<"merge-cast-chain">
{
    let summary = "merge the chain of cast operations into one cast";

    let constructor = "mlir::emitHLS::createEmitHLSMergeCastChainPass()";
}

#endif //EMITHLS_MLIR_TRANSFORM_PASSES
