/// Implementation of OperatorToProcess transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Dialect/emitHLS/Transforms/MergeCastChain.h"

#include "dfg-mlir/Dialect/emitHLS/IR/Dialect.h"
#include "dfg-mlir/Dialect/emitHLS/IR/Ops.h"
#include "dfg-mlir/Dialect/emitHLS/Transforms/Passes.h"
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
namespace emitHLS {
#define GEN_PASS_DEF_EMITHLSMERGECASTCHAIN
#include "dfg-mlir/Dialect/emitHLS/Transforms/Passes.h.inc"
} // namespace emitHLS
} // namespace mlir

using namespace mlir;
using namespace emitHLS;

namespace {
struct ReplaceCastValue : public OpRewritePattern<ArithCastOp> {
    ReplaceCastValue(MLIRContext* context)
            : OpRewritePattern<ArithCastOp>(context){};

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

void mlir::emitHLS::populateMergeCastChainConversionPatterns(
    RewritePatternSet &patterns)
{
    patterns.add<ReplaceCastValue>(patterns.getContext());
}

namespace {
struct emitHLSMergeCastChainPass
        : public emitHLS::impl::emitHLSMergeCastChainBase<
              emitHLSMergeCastChainPass> {
    void runOnOperation() override;
};
} // namespace

void emitHLSMergeCastChainPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateMergeCastChainConversionPatterns(patterns);

    target.addLegalDialect<emitHLSDialect>();
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

std::unique_ptr<Pass> mlir::emitHLS::createEmitHLSMergeCastChainPass()
{
    return std::make_unique<emitHLSMergeCastChainPass>();
}
