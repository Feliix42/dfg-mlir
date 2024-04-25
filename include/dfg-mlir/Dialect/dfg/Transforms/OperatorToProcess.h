/// Declaration of dfg operator to process transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace dfg {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_DFGOPERATORTOPROCESS
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateOperatorToProcessConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createDfgOperatorToProcessPass();

} // namespace dfg
} // namespace mlir
