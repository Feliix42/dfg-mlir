/// Implementation of the dfg-verify-fusion-groups transform pass.
///
/// Checks all `!dfg.fusion` groups for legality (same checks as
/// dfg-fuse-groups) and reports hard errors. Useful for validating
/// annotations without fusing.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Dialect/dfg/Transforms/FusionGroups.h"

#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h"

namespace mlir {
namespace dfg {
#define GEN_PASS_DEF_DFGVERIFYFUSIONGROUPS
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"
} // namespace dfg
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;

namespace {
struct DfgVerifyFusionGroupsPass
        : public dfg::impl::DfgVerifyFusionGroupsBase<DfgVerifyFusionGroupsPass> {
    void runOnOperation() override;
};
} // namespace

void DfgVerifyFusionGroupsPass::runOnOperation()
{
    Operation *op = getOperation();
    auto groups = collectFusionGroups(op);
    bool ok = true;
    for (FusionGroup &g : groups) {
        auto err = [&](Operation *site, const Twine &msg) {
            site->emitError(msg);
            ok = false;
        };
        (void)analyzeFusionGroup(g, err);
    }
    if (!ok)
        signalPassFailure();
}

std::unique_ptr<Pass> mlir::dfg::createDfgVerifyFusionGroupsPass()
{
    return std::make_unique<DfgVerifyFusionGroupsPass>();
}
