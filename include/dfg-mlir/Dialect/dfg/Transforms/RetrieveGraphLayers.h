/// Declaration of retrieve graph layers transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace dfg {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_DFGRETRIEVEGRAPHLAYERS
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateRetrieveGraphLayersConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createDfgRetrieveGraphLayersPass();

} // namespace dfg
} // namespace mlir
