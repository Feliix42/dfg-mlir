/// Declaration of the Dfg to emitHLS lowering pass that lowers dfg programs
/// to the emitHLS HLS dialect.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTDFGTOEMITHLS
#include "dfg-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateDfgToEmitHLSConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertDfgToEmitHLSPass();

namespace dfg {
void registerConvertToEmitHLSPipelines();
void addConvertToEmitHLSPasses(OpPassManager &pm);
} // namespace dfg

} // namespace mlir
