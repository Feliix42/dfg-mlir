/// Declaration of bufferization transformation.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace dfg {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_DFGBUFFERIZE
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateBufferizePatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createDfgBufferizePass();

} // namespace dfg
} // namespace mlir
