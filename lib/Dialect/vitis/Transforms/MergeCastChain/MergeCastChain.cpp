/// Implementation of OperatorToProcess transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Dialect/vitis/IR/Dialect.h"
#include "dfg-mlir/Dialect/vitis/IR/Ops.h"
#include "dfg-mlir/Dialect/vitis/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/ValueRange.h>

namespace mlir {
namespace vitis {
#define GEN_PASS_DEF_VITISMERGECASTCHAIN
#include "dfg-mlir/Dialect/vitis/Transforms/Passes.h.inc"
} // namespace vitis
} // namespace mlir

using namespace mlir;
using namespace vitis;

namespace {
struct ReplaceCastValue : public OpRewritePattern<ArithCastOp> {
    ReplaceCastValue(MLIRContext* context)
            : OpRewritePattern<ArithCastOp>(context) {};

    LogicalResult
    matchAndRewrite(ArithCastOp op, PatternRewriter &rewriter) const override
    {
        auto operand = op.getOperand();
        auto definingOpOperand =
            dyn_cast<ArithCastOp>(operand.getDefiningOp()).getOperand();
        op->replaceUsesOfWith(operand, definingOpOperand);
        return success();
    }
};
} // namespace

void mlir::vitis::populateMergeCastChainConversionPatterns(
    RewritePatternSet &patterns)
{
    patterns.add<ReplaceCastValue>(patterns.getContext());
}

namespace {
struct VitisMergeCastChainPass
        : public vitis::impl::VitisMergeCastChainBase<VitisMergeCastChainPass> {
    void runOnOperation() override;
};
} // namespace

void VitisMergeCastChainPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateMergeCastChainConversionPatterns(patterns);

    target.addLegalDialect<VitisDialect>();
    target.addDynamicallyLegalOp<ArithCastOp>([](ArithCastOp castOp) {
        if (isa<ArithCastOp>(castOp.getOperand().getDefiningOp())) return false;
        return true;
    });
    target.markUnknownOpDynamicallyLegal([](Operation* op) { return true; });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::vitis::createVitisMergeCastChainPass()
{
    return std::make_unique<VitisMergeCastChainPass>();
}
