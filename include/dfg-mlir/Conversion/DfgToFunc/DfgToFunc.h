/// Declaration of the Dfg to Func lowering pass that lowers graph nodes to func
/// declarations
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#pragma once

#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

//===- Generated includes -------------------------------------------------===//

// NOTE: must be included at global scope (the generated file opens its own
// `namespace mlir` / `namespace llvm` blocks)
#include "dfg-mlir/Conversion/PassesEnums.h.inc"

namespace mlir {

#define GEN_PASS_DECL_CONVERTDFGTOFUNC
#include "dfg-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

/// Strategy interface for spawning the nodes of a dataflow graph in parallel.
///
/// The `convert-dfg-nodes-to-func` pass is parametric in the parallel backend
/// (see the `parallel-backend` option): the backend-agnostic parts of the
/// lowering (processes to functions, loops to cf) are shared, while the
/// emission of the parallel instantiation of the graph nodes is delegated to
/// a SpawnStrategy.
class SpawnStrategy {
public:
    virtual ~SpawnStrategy() = default;

    /// Emit the parallel execution of a group of successive
    /// `dfg.instantiate` operations. The insertion point of `builder` is at
    /// the first operation of the group; the group's instantiations are
    /// consumed (moved/erased) by the emission.
    virtual void emitParallelGroup(OpBuilder &builder, Location loc,
                                   ArrayRef<dfg::InstantiateOp> group) const = 0;

    /// Lower a single `dfg.instantiate` (inside a parallel group) to its
    /// backend-specific invocation.
    virtual LogicalResult
    emitInstantiation(dfg::InstantiateOp op, dfg::InstantiateOpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter) const = 0;
};

/// Create the spawn strategy for the given parallel backend.
std::unique_ptr<SpawnStrategy> createSpawnStrategy(ParallelBackend backend);

void populateDfgToFuncConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns,
    const SpawnStrategy &spawnStrategy);

std::unique_ptr<Pass> createConvertDfgToFuncPass();
std::unique_ptr<Pass> createConvertDfgToFuncPass(ConvertDfgToFuncOptions options);

} // namespace mlir
