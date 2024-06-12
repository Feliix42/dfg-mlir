/// Declares the dfg DataflowGraphOp interface.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/IR/OpDefinition.h"

#include "llvm/ADT/GraphTraits.h"

namespace mlir::dfg {

// Verifies a graph's content is correct
// LogicalResult verify(Operation* op);

// template <>
// struct llvm::GraphTraits<>

} // namespace mlir::dfg

//===- Generated includes -------------------------------------------------===//

#include "dfg-mlir/Dialect/dfg/Interfaces/DataflowGraphOp.h.inc"

//===----------------------------------------------------------------------===//
