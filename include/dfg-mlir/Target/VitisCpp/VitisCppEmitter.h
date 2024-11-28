/// Declaration of the vitis cpp emitter.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace vitis {

/// Translates the given operation to C++ code. The operation or operations in
/// the region of 'op' need almost all be in Vitis dialect.
LogicalResult translateToVitisCpp(Operation* op, raw_ostream &os);

} // namespace vitis
} // namespace mlir
