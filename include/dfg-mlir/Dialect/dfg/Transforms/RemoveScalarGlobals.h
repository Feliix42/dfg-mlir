/// Declaration of removeing scalar memref globals transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace dfg {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_DFGREMOVESCALARGLOBALS
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateRemoveScalarGlobalsConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createDfgRemoveScalarGlobalsPass();

} // namespace dfg
} // namespace mlir
