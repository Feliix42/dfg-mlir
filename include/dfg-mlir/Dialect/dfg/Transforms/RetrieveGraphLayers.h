/// Declaration of retrieve graph layers transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace func {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_FUNCRETRIEVEGRAPHLAYERS
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateRetrieveGraphLayersConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createFuncRetrieveGraphLayersPass();

} // namespace func
} // namespace mlir
