/// Declaration of vitis insert includes transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace vitis {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_VITISINSERTINCLUDES
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createVitisInsertIncludesPass();

} // namespace vitis
} // namespace mlir
