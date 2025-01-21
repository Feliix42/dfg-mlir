/// Declaration of lower inside to scf level transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace dfg {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_DFGLOWERINSIDETOLINALG
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

// void populateLowerInsideToScfConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createDfgLowerInsideToLinalgPass();

void registerDfgLowerInsideToScfPipelines();

void addDfgLowerInsideToScfPasses(OpPassManager &pm);

} // namespace dfg
} // namespace mlir
