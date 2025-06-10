/// Declaration of the Math to emitHLS lowering pass that lowers those
/// dialects to the emitHLS HLS dialect.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTMATHTOEMITHLS
#include "dfg-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateMathToEmitHLSConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertMathToEmitHLSPass();

} // namespace mlir
