/// Implementation of OperatorToProcess transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Dialect/dfg/Transforms/OperatorToProcess.h"

#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/ValueRange.h>

namespace mlir {
namespace dfg {
#define GEN_PASS_DEF_DFGOPERATORTOPROCESS
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"
} // namespace dfg
} // namespace mlir

using namespace mlir;
using namespace dfg;

namespace {

struct ConvertAnyOperatorToEquivalentProcess
        : public OpRewritePattern<OperatorOp> {
    ConvertAnyOperatorToEquivalentProcess(MLIRContext* context)
            : OpRewritePattern<OperatorOp>(context){};

    LogicalResult
    matchAndRewrite(OperatorOp op, PatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto funcTy = op.getFunctionType();
        auto numInputs = funcTy.getNumInputs();
        auto numOutputs = funcTy.getNumResults();

        SmallVector<Type> inChanTypes, outChanTypes;
        for (const auto inTy : funcTy.getInputs())
            inChanTypes.push_back(OutputType::get(rewriter.getContext(), inTy));
        for (const auto outTy : funcTy.getResults())
            outChanTypes.push_back(
                InputType::get(rewriter.getContext(), outTy));
        auto newFuncTy = rewriter.getFunctionType(inChanTypes, outChanTypes);

        // Creating new process op
        auto processOp =
            rewriter.create<ProcessOp>(loc, op.getSymNameAttr(), newFuncTy);
        Block* processBlock = &processOp.getBody().front();
        IRMapping mapper;
        for (auto [oldArg, newArg] : llvm::zip(
                 op.getBody().getArguments(),
                 processOp.getBody().getArguments()))
            mapper.map(oldArg, newArg);

        rewriter.setInsertionPointToEnd(processBlock);
        SmallVector<Value> iterArgs;
        for (auto &opi : op.getInitBody().getOps()) {
            if (isa<YieldOp>(opi)) {
                auto yieldCloned = rewriter.clone(opi, mapper);
                DenseMap<Value, unsigned> valueToIndex;
                for (unsigned i = 0, e = yieldCloned->getNumOperands(); i < e;
                     ++i)
                    valueToIndex[yieldCloned->getOperand(i)] = i;
                llvm::sort(iterArgs, [&](Value a, Value b) {
                    return valueToIndex[a] < valueToIndex[b];
                });
                rewriter.eraseOp(yieldCloned);
                break;
            }
            auto opCloned = rewriter.clone(opi, mapper);
            iterArgs.push_back(opCloned->getResult(0));
        }
        // In process, create new operator op
        auto loopOp = rewriter.create<LoopOp>(
            loc,
            processOp.getBody().getArguments().take_front(numInputs),
            processOp.getBody().getArguments().take_back(numOutputs),
            iterArgs);
        Block* loopBlock = &loopOp.getBody().front();
        rewriter.setInsertionPointToEnd(loopBlock);
        // Create pull ops for input channels
        for (size_t i = 0; i < numInputs; i++) {
            auto pullOp =
                rewriter.create<PullOp>(loc, processBlock->getArgument(i));
            mapper.map(op.getBody().getArgument(i), pullOp.getResult());
        }
        // Copy the content into loop
        for (auto &opi : op.getBody().getOps()) {
            // Expand output op into push ops
            if (isa<OutputOp>(opi)) {
                auto newOp = rewriter.clone(opi, mapper);
                for (auto [pushedValue, OutputChannel] : llvm::zip(
                         newOp->getOperands(),
                         processOp.getBody()
                             .getArguments()
                             .drop_front(numInputs)
                             .take_front(numOutputs))) {
                    rewriter.create<PushOp>(loc, pushedValue, OutputChannel);
                }
                rewriter.eraseOp(newOp);
                break;
            } else {
                if (isa<arith::ConstantOp>(opi))
                    rewriter.setInsertionPoint(loopOp);
                rewriter.clone(opi, mapper);
                rewriter.setInsertionPointToEnd(loopBlock);
            }
        }

        rewriter.eraseOp(op);
        return success();
    }
};
} // namespace

void mlir::dfg::populateOperatorToProcessConversionPatterns(
    RewritePatternSet &patterns)
{
    patterns.add<ConvertAnyOperatorToEquivalentProcess>(patterns.getContext());
}

namespace {
struct DfgOperatorToProcessPass
        : public dfg::impl::DfgOperatorToProcessBase<DfgOperatorToProcessPass> {
    void runOnOperation() override;
};
} // namespace

void DfgOperatorToProcessPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateOperatorToProcessConversionPatterns(patterns);

    target.addLegalDialect<DfgDialect>();
    target.addIllegalOp<OperatorOp>();
    target.markUnknownOpDynamicallyLegal([](Operation* op) {
        return op->getDialect()->getNamespace() != "dfg";
    });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::dfg::createDfgOperatorToProcessPass()
{
    return std::make_unique<DfgOperatorToProcessPass>();
}
