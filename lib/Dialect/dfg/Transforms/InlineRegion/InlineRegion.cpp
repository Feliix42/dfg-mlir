/// Implementation of InlineRegion transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using RegionEmbedMap = DenseMap<Operation*, SmallVector<Operation*>>;
using ValueMap = DenseMap<Value, Value>;

namespace mlir {
namespace dfg {
#define GEN_PASS_DEF_DFGINLINEREGION
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"
} // namespace dfg
} // namespace mlir

using namespace mlir;
using namespace dfg;
using namespace llvm;

namespace {
struct FullyInlinePattern : public OpRewritePattern<EmbedOp> {
    using OpRewritePattern<EmbedOp>::OpRewritePattern;

    FullyInlinePattern(MLIRContext* context, RegionEmbedMap &regionEmbedMap)
            : OpRewritePattern<EmbedOp>(context),
              regionEmbedMap(regionEmbedMap)
    {}

    LogicalResult
    matchAndRewrite(EmbedOp embedOp, PatternRewriter &rewriter) const override
    {
        SymbolRefAttr regionRef = embedOp.getCallee();
        auto regionOp =
            SymbolTable::lookupNearestSymbolFrom<RegionOp>(embedOp, regionRef);

        ValueMap oldToNewValues;
        for (auto [regionArg, embedArg] : llvm::zip(
                 regionOp.getBody().getArguments(),
                 embedOp->getOperands())) {
            oldToNewValues[regionArg] = embedArg;
        }

        rewriter.setInsertionPoint(embedOp);
        for (Operation &op : regionOp.getBody().getOps()) {
            Operation* newOp = rewriter.clone(op);
            for (size_t i = 0; i < newOp->getResults().size(); i++)
                oldToNewValues[op.getResult(i)] = newOp->getResult(i);
            for (size_t i = 0; i < newOp->getOperands().size(); i++) {
                auto operand = newOp->getOperand(i);
                if (oldToNewValues.count(operand))
                    newOp->setOperand(i, oldToNewValues[operand]);
            }
        }

        rewriter.eraseOp(embedOp);

        if (regionOp->use_empty()) {
            regionEmbedMap[regionOp].clear();
            rewriter.eraseOp(regionOp);
        }

        return success();
    }

private:
    RegionEmbedMap &regionEmbedMap;
};
} // namespace

namespace {
struct DfgInlineRegionPass
        : public dfg::impl::DfgInlineRegionBase<DfgInlineRegionPass> {
    void runOnOperation() override;

private:
    void applyFullInliningPatterns()
    {
        Operation* op = getOperation();
        RegionEmbedMap regionEmbedMap;
        collectRegionToEmbedMapping(op, regionEmbedMap);

        RewritePatternSet patterns(&getContext());
        patterns.add<FullyInlinePattern>(&getContext(), regionEmbedMap);

        ConversionTarget target(getContext());
        target.addIllegalOp<EmbedOp>();
        target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

        if (failed(applyPartialConversion(op, target, std::move(patterns))))
            signalPassFailure();
    }

    void applySmartInliningPatterns()
    {
        // Smart strategy will be implemented later
    }

    void collectRegionToEmbedMapping(
        Operation* module,
        RegionEmbedMap regionEmbedMap)
    {
        module->walk([&](Operation* op) {
            if (auto embedOp = dyn_cast<EmbedOp>(op)) {
                SymbolRefAttr callee = embedOp.getCallee();
                auto regionOp = SymbolTable::lookupNearestSymbolFrom<RegionOp>(
                    embedOp,
                    callee);
                regionEmbedMap[regionOp].push_back(embedOp);
            }
        });
    }
};
} // namespace

void DfgInlineRegionPass::runOnOperation()
{
    std::string strategyOption = strategy;
    if (strategyOption != "full" && strategyOption != "smart") {
        emitError(
            UnknownLoc::get(&getContext()),
            "Invalid strategy option: " + strategyOption
                + ". Expected 'full' or 'smart'.");
        signalPassFailure();
        return;
    }

    if (strategyOption == "full") {
        // Implement full inlining strategy
        applyFullInliningPatterns();
    } else if (strategyOption == "smart") {
        // Implement smart inlining strategy
        applySmartInliningPatterns();
    }
}

std::unique_ptr<Pass> mlir::dfg::createDfgInlineRegionPass()
{
    return std::make_unique<DfgInlineRegionPass>();
}
