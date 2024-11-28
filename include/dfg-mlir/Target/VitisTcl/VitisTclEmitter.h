/// Declaration of the vitis tcl emitter.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace vitis {

/// Translates the given operation to tcl script. The operation or operations in
/// the region of 'op' need almost all be in Vitis dialect.
LogicalResult
translateToVitisTcl(Operation* op, raw_ostream &os, std::string &targetDevice);

} // namespace vitis
} // namespace mlir
