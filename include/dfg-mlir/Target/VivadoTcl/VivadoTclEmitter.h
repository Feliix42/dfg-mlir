/// Declaration of the vivado tcl emitter.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace dfg {

/// Translates the top region to tcl script.
LogicalResult
translateToVivadoTcl(Operation* op, raw_ostream &os, std::string &targetDevice);

} // namespace dfg
} // namespace mlir
