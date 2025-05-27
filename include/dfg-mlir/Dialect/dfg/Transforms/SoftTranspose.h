/// Declaration of soft transpose transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace linalg {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_LINALGSOFTTRANSPOSE
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateSoftTransposeConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createLinalgSoftTransposePass();

} // namespace linalg
} // namespace mlir
