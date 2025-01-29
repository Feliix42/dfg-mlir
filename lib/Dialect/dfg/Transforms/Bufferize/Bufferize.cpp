/// Implements the dfg dialect ops bufferization.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Dialect/dfg/Transforms/Bufferize/Bufferize.h"

#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <new>

namespace mlir {
namespace dfg {
#define GEN_PASS_DEF_DFGBUFFERIZE
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"
} // namespace dfg
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;

namespace {

template<typename OpT>
struct BufferizeProcessRegion : public OpRewritePattern<OpT> {
    BufferizeProcessRegion(MLIRContext* context)
            : OpRewritePattern<OpT>(context) {};

    LogicalResult
    matchAndRewrite(OpT op, PatternRewriter &rewriter) const override
    {
        auto funcTy = op.getFunctionType();
        SmallVector<Type> inputTypes, outputTypes;
        for (auto inputType : funcTy.getInputs()) {
            auto elemTy = dyn_cast<OutputType>(inputType).getElementType();
            if (auto tensorTy = dyn_cast<TensorType>(elemTy)) {
                // If it's tensor type, convert it to memref type.
                auto memrefTy = MemRefType::get(
                    tensorTy.getShape(),
                    tensorTy.getElementType());
                inputTypes.push_back(
                    OutputType::get(tensorTy.getContext(), memrefTy));
            } else {
                // If it's not tensor, keep the type.
                inputTypes.push_back(inputType);
            }
        }
        for (auto outputType : funcTy.getResults()) {
            auto elemTy = dyn_cast<InputType>(outputType).getElementType();
            if (auto tensorTy = dyn_cast<TensorType>(elemTy)) {
                // If it's tensor type, convert it to memref type.
                auto memrefTy = MemRefType::get(
                    tensorTy.getShape(),
                    tensorTy.getElementType());
                outputTypes.push_back(
                    InputType::get(tensorTy.getContext(), memrefTy));
            } else {
                // If it's not tensor, remain the type.
                outputTypes.push_back(outputType);
            }
        }

        auto newOp = rewriter.create<OpT>(
            op.getLoc(),
            op.getSymNameAttr(),
            rewriter.getFunctionType(inputTypes, outputTypes));
        Block* processBlock = &newOp.getBody().front();
        IRMapping mapper;
        for (auto [oldArg, newArg] : llvm::zip(
                 op.getBody().getArguments(),
                 newOp.getBody().getArguments()))
            mapper.map(oldArg, newArg);
        rewriter.setInsertionPointToEnd(processBlock);
        for (auto &opi : op.getBody().getOps()) rewriter.clone(opi, mapper);

        rewriter.eraseOp(op);
        return success();
    }
};

struct BufferizePull : public OpRewritePattern<PullOp> {
    BufferizePull(MLIRContext* context) : OpRewritePattern<PullOp>(context) {};

    LogicalResult
    matchAndRewrite(PullOp op, PatternRewriter &rewriter) const override
    {
        auto tensorTy = dyn_cast<TensorType>(op.getOutp().getType());
        auto memrefTy =
            MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
        auto newPull =
            rewriter.create<PullOp>(op.getLoc(), memrefTy, op.getChan());

        auto toTensorOp = rewriter.create<bufferization::ToTensorOp>(
            op.getLoc(),
            newPull.getResult(),
            true);
        op.getOutp().replaceAllUsesWith(toTensorOp.getResult());

        rewriter.replaceOp(op, newPull);
        return success();
    }
};

struct BufferizePush : public OpRewritePattern<PushOp> {
    BufferizePush(MLIRContext* context) : OpRewritePattern<PushOp>(context) {};

    LogicalResult
    matchAndRewrite(PushOp op, PatternRewriter &rewriter) const override
    {
        auto tensor = op.getInp();
        auto memrefTy = op.getChan().getType().getElementType();
        rewriter.setInsertionPoint(op);
        auto toMemrefType = rewriter.create<bufferization::ToMemrefOp>(
            op.getLoc(),
            memrefTy,
            tensor);
        op->setOperand(0, toMemrefType.getResult());

        return success();
    }
};

struct BufferizeChannel : public OpRewritePattern<ChannelOp> {
    BufferizeChannel(MLIRContext* context)
            : OpRewritePattern<ChannelOp>(context) {};

    LogicalResult
    matchAndRewrite(ChannelOp op, PatternRewriter &rewriter) const override
    {
        auto tensorTy = dyn_cast<TensorType>(op.getEncapsulatedType());
        auto memrefTy =
            MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
        auto newChannel = rewriter.create<ChannelOp>(
            op.getLoc(),
            memrefTy,
            op.getBufferSize().has_value() ? op.getBufferSize().value() : 0);
        op.getResult(0).replaceAllUsesWith(newChannel.getResult(0));
        op.getResult(1).replaceAllUsesWith(newChannel.getResult(1));

        rewriter.replaceOp(op, newChannel);
        return success();
    }
};

} // namespace

void mlir::dfg::populateBufferizePatterns(RewritePatternSet &patterns)
{
    patterns.add<BufferizeProcessRegion<ProcessOp>>(patterns.getContext());
    patterns.add<BufferizePull>(patterns.getContext());
    patterns.add<BufferizePush>(patterns.getContext());
    patterns.add<BufferizeProcessRegion<RegionOp>>(patterns.getContext());
    patterns.add<BufferizeChannel>(patterns.getContext());
}

namespace {
struct DfgBufferizePass : public dfg::impl::DfgBufferizeBase<DfgBufferizePass> {
    void runOnOperation() override;
};
} // namespace

void DfgBufferizePass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateBufferizePatterns(patterns);

    target.addLegalDialect<DfgDialect, bufferization::BufferizationDialect>();
    target.addDynamicallyLegalOp<ProcessOp>([](ProcessOp op) {
        auto funcTy = op.getFunctionType();
        for (auto inputType : funcTy.getInputs())
            if (isa<TensorType>(
                    dyn_cast<OutputType>(inputType).getElementType()))
                return false;
        for (auto outputType : funcTy.getResults())
            if (isa<TensorType>(
                    dyn_cast<InputType>(outputType).getElementType()))
                return false;
        return true;
    });
    target.addDynamicallyLegalOp<PullOp>([](PullOp op) {
        if (isa<TensorType>(op.getOutp().getType())) return false;
        return true;
    });
    target.addDynamicallyLegalOp<PushOp>([](PushOp op) {
        if (isa<TensorType>(op.getInp().getType())) return false;
        return true;
    });
    target.addDynamicallyLegalOp<RegionOp>([](RegionOp op) {
        auto funcTy = op.getFunctionType();
        for (auto inputType : funcTy.getInputs())
            if (isa<TensorType>(
                    dyn_cast<OutputType>(inputType).getElementType()))
                return false;
        for (auto outputType : funcTy.getResults())
            if (isa<TensorType>(
                    dyn_cast<InputType>(outputType).getElementType()))
                return false;
        return true;
    });
    target.addDynamicallyLegalOp<ChannelOp>([](ChannelOp op) {
        if (isa<TensorType>(op.getEncapsulatedType())) return false;
        return true;
    });
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::dfg::createDfgBufferizePass()
{
    return std::make_unique<DfgBufferizePass>();
}
