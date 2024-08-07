/// Implementation of OperatorToProcess transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace dfg {
#define GEN_PASS_DEF_DFGOPERATORTOPROCESS
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"
} // namespace dfg
} // namespace mlir

using namespace mlir;
using namespace dfg;

namespace {

// struct ExpandYieldToPushes : public OpRewritePattern<YieldOp> {
//     ExpandYieldToPushes(MLIRContext* context)
//             : OpRewritePattern<YieldOp>(context){};

//     LogicalResult
//     matchAndRewrite(YieldOp op, PatternRewriter &rewriter) const override
//     {
//         auto loc = op.getLoc();
//         OperatorOp operatorOp = op.getParentOp<OperatorOp>();
//         auto idxOutChans = operatorOp.getFunctionType().getNumInputs();

//         for (const auto operand : op.getOperands()) {
//             rewriter.create<PushOp>(
//                 loc,
//                 operand,
//                 operatorOp.getBody().getArgument(idxOutChans++));
//         }
//         rewriter.eraseOp(op);

//         return success();
//     }
// };

std::optional<int> getNumBlockArg(SmallVector<Value> list, Value value)
{
    for (size_t i = 0; i < list.size(); i++)
        if (list[i] == value) return i;

    return std::nullopt;
}

SmallVector<std::pair<Value, Value>> oldToNewValueMap, oldToNewIterArgs;
void processNestedRegion(
    Operation* op,
    SmallVector<Value> src,
    SmallVector<Value> dest)
{
    for (size_t i = 0; i < op->getNumOperands(); i++) {
        // TODO: add new map from old iter_args to new one.
        auto operand = op->getOperand(i);
        auto numOperand = getNumBlockArg(src, operand);
        auto newResultValue =
            getNewIndexOrArg<Value, Value>(operand, oldToNewValueMap);
        auto newIterArgValue =
            getNewIndexOrArg<Value, Value>(operand, oldToNewIterArgs);
        if (numOperand.has_value()) op->setOperand(i, dest[numOperand.value()]);
        if (newResultValue.has_value())
            op->setOperand(i, newResultValue.value());
        if (newIterArgValue.has_value())
            op->setOperand(i, newIterArgValue.value());
    }
    if (op->getNumRegions() != 0)
        for (auto &region : op->getRegions())
            for (auto &opi : region.getOps())
                processNestedRegion(&opi, src, dest);
}

void reorderIterArgs(
    SmallVector<Value> &values,
    SmallVector<Value> origin,
    YieldOp yieldOp)
{
    llvm::DenseMap<Value, unsigned> valueIndexMap;
    for (unsigned i = 0; i < origin.size(); i++) valueIndexMap[origin[i]] = i;

    SmallVector<Value> reorderedValues(values.size());
    for (unsigned i = 0; i < yieldOp.getNumOperands(); i++) {
        Value operand = yieldOp.getOperand(i);
        assert(valueIndexMap.count(operand) && "Operand not found in values");
        reorderedValues[i] = values[valueIndexMap[operand]];
    }
    values = reorderedValues;
}

struct ConvertAnyOperatorToEquivalentProcess
        : public OpRewritePattern<OperatorOp> {
    ConvertAnyOperatorToEquivalentProcess(MLIRContext* context)
            : OpRewritePattern<OperatorOp>(context){};

    LogicalResult
    matchAndRewrite(OperatorOp op, PatternRewriter &rewriter) const override
    {

        auto loc = op.getLoc();
        auto name = op.getSymNameAttr();
        auto funcTy = op.getFunctionType();
        auto bias = funcTy.getNumInputs() + funcTy.getNumResults();
        auto initValueOps = op.getInitBody().getOps();
        SmallVector<Value> blockArgs, oldIterArgs;
        blockArgs.append(
            op.getBody().getArguments().begin(),
            op.getBody().getArguments().begin() + bias);
        oldIterArgs.append(
            op.getBody().getArguments().begin() + bias,
            op.getBody().getArguments().end());

        SmallVector<Type> inChanTypes, outChanTypes;
        for (const auto inTy : funcTy.getInputs())
            inChanTypes.push_back(OutputType::get(rewriter.getContext(), inTy));
        for (const auto outTy : funcTy.getResults())
            outChanTypes.push_back(
                InputType::get(rewriter.getContext(), outTy));

        // Create the new ProcessOp
        auto processOp = rewriter.create<ProcessOp>(
            loc,
            name,
            FunctionType::get(
                rewriter.getContext(),
                inChanTypes,
                outChanTypes));
        auto newFuncTy = processOp.getFunctionType();
        Block* processBlock = &processOp.getBody().front();
        rewriter.setInsertionPointToEnd(processBlock);
        SmallVector<Value> iterArgs, initConstants;
        for (auto &opi : initValueOps) {
            if (auto yieldOp = dyn_cast<YieldOp>(opi)) {
                reorderIterArgs(iterArgs, initConstants, yieldOp);
                break;
            }
            auto newOpi = opi.clone();
            rewriter.insert(newOpi);
            initConstants.push_back(opi.getResult(0));
            iterArgs.push_back(newOpi->getResult(0));
        }
        for (auto iterArg : llvm::zip(iterArgs, oldIterArgs)) {
            oldToNewIterArgs.push_back(
                std::make_pair(std::get<0>(iterArg), std::get<1>(iterArg)));
        }

        // Insert LoopOp in the ProcessOp
        rewriter.setInsertionPointToEnd(processBlock);
        SmallVector<Value> inChans, outChans;
        for (size_t i = 0; i < newFuncTy.getNumInputs(); i++)
            inChans.push_back(processBlock->getArgument(i));
        auto loopOp = rewriter.create<LoopOp>(loc, inChans, outChans, iterArgs);

        // Insert number of input channels PullOps
        rewriter.setInsertionPointToStart(&loopOp.getBody().front());
        SmallVector<Value> newOperands;
        SmallVector<std::pair<Value, Value>> argToPulledMap;
        for (size_t i = 0; i < newFuncTy.getNumInputs(); i++) {
            auto pullOp =
                rewriter.create<PullOp>(loc, processBlock->getArgument(i));
            newOperands.push_back(pullOp.getResult());
            argToPulledMap.push_back(std::make_pair(
                pullOp.getResult(),
                op.getBody().getArgument(i)));
        }

        for (auto &opi : op.getBody().getOps()) {
            if (!isa<OutputOp>(opi)) {
                // Insert original ops into LoopOp and replace the operand
                auto newOpi = opi.clone();
                processNestedRegion(newOpi, blockArgs, newOperands);
                rewriter.insert(newOpi);
                for (size_t i = 0; i < opi.getNumResults(); i++)
                    oldToNewValueMap.push_back(
                        std::make_pair(newOpi->getResult(i), opi.getResult(i)));
            } else {
                // Replace the OutputOp with number of output channels PushOps
                auto outputOp = dyn_cast<OutputOp>(opi);
                auto idxBias = newFuncTy.getNumInputs();
                for (const auto operand : outputOp.getOperands())
                    if (auto newValue = getNewIndexOrArg<Value, Value>(
                            operand,
                            oldToNewValueMap))
                        rewriter.create<PushOp>(
                            loc,
                            newValue.value(),
                            processBlock->getArgument(idxBias++));
                    else if (
                        auto newValue = getNewIndexOrArg<Value, Value>(
                            operand,
                            argToPulledMap))
                        rewriter.create<PushOp>(
                            loc,
                            newValue.value(),
                            processBlock->getArgument(idxBias++));
                    else
                        return rewriter.notifyMatchFailure(
                            outputOp.getLoc(),
                            "Unknown value to be pushed.");
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
    // patterns.add<ExpandYieldToPushes>(patterns.getContext());
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
