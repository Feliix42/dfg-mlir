/// Declaration of dfg inline region transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace dfg {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_DFGINLINEREGION
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateInlineRegionConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createDfgInlineRegionPass();

} // namespace dfg
} // namespace mlir
