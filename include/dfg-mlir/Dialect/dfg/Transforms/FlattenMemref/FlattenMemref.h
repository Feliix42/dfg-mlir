/// Declaration of flatten memref transformation.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace dfg {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_DFGFLATTENMEMREF
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateFlattenMemrefPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createDfgFlattenMemrefPass();

} // namespace dfg
} // namespace mlir
