/// Declares the conversion passes.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "dfg-mlir/Dialect/vitis/Transforms/InsertIncludes.h"
#include "dfg-mlir/Dialect/vitis/Transforms/MergeCastChain.h"

namespace mlir::vitis {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "dfg-mlir/Dialect/vitis/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir::vitis
