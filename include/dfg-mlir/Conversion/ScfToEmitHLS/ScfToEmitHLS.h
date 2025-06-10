/// Declaration of the SCF to emitHLS lowering pass that lowers memref
/// operations to the emitHLS HLS dialect.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTSCFTOEMITHLS
#include "dfg-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateScfToEmitHLSConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertScfToEmitHLSPass();

} // namespace mlir
