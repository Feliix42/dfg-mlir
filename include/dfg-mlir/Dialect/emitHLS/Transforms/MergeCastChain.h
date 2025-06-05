/// Declaration of emitHLS merge cast chain transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace emitHLS {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_EMITHLSMERGECASTCHAIN
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateMergeCastChainConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createEmitHLSMergeCastChainPass();

} // namespace emitHLS
} // namespace mlir
