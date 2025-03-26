/// Declaration of vitis merge cast chain transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace vitis {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_VITISMERGECASTCHAIN
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateMergeCastChainConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createVitisMergeCastChainPass();

} // namespace vitis
} // namespace mlir
