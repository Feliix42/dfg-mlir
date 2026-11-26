/// Shared analysis for fusion-group passes (dfg-fuse-groups,
/// dfg-verify-fusion-groups).
///
/// A fusion group is the set of `dfg.instantiate` ops in one parent region
/// that carry the same `!dfg.fusion<"name">` annotation. This file collects
/// those groups, classifies the channels between them (internal vs. external)
/// and checks legality of fusing a group into a single node.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#pragma once

#include "dfg-mlir/Dialect/dfg/IR/Attrs.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace dfg {

struct FusionGroup {
    StringRef name;
    Block *parent = nullptr;
    SmallVector<InstantiateOp> members;

    /// Internal channels (both ends used only by group members), as
    /// (push-side value -> pull-side value) pairs of the channel op.
    SmallVector<std::pair<Value, Value>> internalChans;
    /// Producer -> consumer member indices over internal channels.
    SmallVector<std::pair<unsigned, unsigned>> edges;
    /// Members that have at least one external input channel.
    SmallVector<unsigned> externalRoots;
};

/// Returns the process op instantiated by `inst`, or nullptr.
ProcessOp getMemberProcess(InstantiateOp inst);

/// The entry block of the process body.
Block *procBodyBlock(ProcessOp proc);

/// Whether the process body is wrapped in a dfg.loop.
bool processHasLoop(ProcessOp proc);

/// Collects all fusion groups under `root`, keyed by (parent block, group).
SmallVector<FusionGroup, 0> collectFusionGroups(Operation *root);

/// Analyzes the group's channel wiring. On success, fills in internalChans,
/// edges and externalRoots. `diag` is invoked with (site, message) for every
/// legality violation.
LogicalResult analyzeFusionGroup(
    FusionGroup &g,
    const function_ref<void(Operation *, const Twine &)> &diag);

/// Determines the head member (the node everything is fused into) via
/// dominance over the internal edge graph. Back edges are included in the
/// graph but dominance handles cycles. Ties are broken deterministically:
/// prefer a member whose process contains an scf.for, then the smallest
/// callee name.
unsigned computeHead(FusionGroup &g, ArrayRef<ProcessOp> procs);

} // namespace dfg
} // namespace mlir
