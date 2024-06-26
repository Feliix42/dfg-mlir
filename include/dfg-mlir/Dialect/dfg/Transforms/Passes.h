/// Declares the conversion passes.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "dfg-mlir/Dialect/dfg/Transforms/OpereatorToProcess/OperatorToProcess.h"
#include "dfg-mlir/Dialect/dfg/Transforms/PrintOperatorToYaml/PrintOperatorToYaml.h"

namespace mlir::dfg {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir::dfg
