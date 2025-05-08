/// Declaration of inline scalar argument transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace linalg {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_LINALGINLINESCALARARGUMENT
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateInlineScalarArgumentConversionPatterns(
    RewritePatternSet &patterns);

std::unique_ptr<Pass> createLinalgInlineScalarArgumentPass();

} // namespace linalg
} // namespace mlir
