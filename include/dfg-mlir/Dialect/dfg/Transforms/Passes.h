/// Declares the conversion passes.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "dfg-mlir/Dialect/dfg/Transforms/InlineRegion.h"
#include "dfg-mlir/Dialect/dfg/Transforms/LowerInsideToScf.h"
#include "dfg-mlir/Dialect/dfg/Transforms/OperatorToProcess.h"
#include "dfg-mlir/Dialect/dfg/Transforms/PrintGraph.h"
#include "dfg-mlir/Dialect/dfg/Transforms/PrintOperatorToYaml.h"
#include "dfg-mlir/Dialect/dfg/Transforms/RetrieveGraphLayers.h"

namespace mlir::dfg {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir::dfg
