/// Declaration of removeing scalar memref globals transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace memref {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_MEMREFREMOVESCALARGLOBALS
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateRemoveScalarGlobalsConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createMemrefRemoveScalarGlobalsPass();

} // namespace memref
} // namespace mlir
