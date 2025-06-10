/// Declaration of emitHLS insert includes transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace emitHLS {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_EMITHLSINSERTINCLUDES
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createEmitHLSInsertIncludesPass();

} // namespace emitHLS
} // namespace mlir
