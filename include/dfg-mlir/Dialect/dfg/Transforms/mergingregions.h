#ifndef DFG_MLIR_DIALECT_DFG_TRANSFORMS_MERGINGREGIONS_H
#define DFG_MLIR_DIALECT_DFG_TRANSFORMS_MERGINGREGIONS_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace dfg {

#define GEN_PASS_DECL_MERGINGREGIONS
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"

// Keep this name consistent with the constructor in Passes.td
std::unique_ptr<Pass> createMergingRegionsPass();

} // namespace dfg
} // namespace mlir

#endif // DFG_MLIR_DIALECT_DFG_TRANSFORMS_MERGINGREGIONS_H