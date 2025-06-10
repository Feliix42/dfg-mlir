/// Declares the conversion passes.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "dfg-mlir/Dialect/emitHLS/Transforms/InsertIncludes.h"
#include "dfg-mlir/Dialect/emitHLS/Transforms/MergeCastChain.h"

namespace mlir::emitHLS {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "dfg-mlir/Dialect/emitHLS/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir::emitHLS
