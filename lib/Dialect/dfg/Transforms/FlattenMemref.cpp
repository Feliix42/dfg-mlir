/// Implements the dfg dialect ops bufferization.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Dialect/dfg/Transforms/FlattenMemref.h"

#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include <numeric>

namespace mlir {
namespace dfg {
#define GEN_PASS_DEF_DFGFLATTENMEMREF
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"
} // namespace dfg
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;

namespace {

void flattenUsers(Operation* op, MemRefType memrefTy, PatternRewriter &rewriter)
{
    auto calculateFlattenedIndex = [&](Operation* op) -> Value {
        auto loc = op->getLoc();
        auto indices = [&]() {
            if (auto loadOp = dyn_cast<memref::LoadOp>(op))
                return loadOp.getIndices();
            else if (auto storeOp = dyn_cast<memref::StoreOp>(op))
                return storeOp.getIndices();
        }();
        // a. create constants based on the shape
        SmallVector<Value> constants;
        for (auto [size, type] :
             llvm::zip(memrefTy.getShape(), indices.getType())) {
            auto constOp = rewriter.create<arith::ConstantOp>(
                loc,
                rewriter.getIntegerAttr(type, size));
            constants.push_back(constOp.getResult());
        }
        // b. calculate the offsets needed to be multiplied with indices
        SmallVector<Value> offsets;
        std::reverse(constants.begin(), constants.end());
        offsets.push_back(constants[0]);
        Value product;
        for (size_t i = 0; i < constants.size() - 1; i++) {
            if (i == 0) {
                product = constants[i];
                continue;
            }
            auto mulOp =
                rewriter.create<arith::MulIOp>(loc, product, constants[i]);
            product = mulOp.getResult();
            offsets.push_back(product);
        }
        // c. calculate the 1D indices
        SmallVector<Value> indexProducts;
        std::reverse(offsets.begin(), offsets.end());
        for (auto [index, offset] : llvm::zip(indices, offsets)) {
            auto mulOp = rewriter.create<arith::MulIOp>(loc, index, offset);
            indexProducts.push_back(mulOp.getResult());
        }
        indexProducts.push_back(indices.back());
        Value index;
        for (size_t i = 0; i < indexProducts.size(); i++) {
            if (i == 0) {
                index = indexProducts[i];
                continue;
            }
            auto addOp =
                rewriter.create<arith::AddIOp>(loc, index, indexProducts[i]);
            index = addOp.getResult();
        }
        return index;
    };
    for (auto user : op->getResult(0).getUsers()) {
        rewriter.setInsertionPoint(user);
        // If it's used in memref.load
        if (auto loadOp = dyn_cast<memref::LoadOp>(user)) {
            // New load with 1D index
            auto newIndex = calculateFlattenedIndex(loadOp);
            auto newLoad = rewriter.create<memref::LoadOp>(
                loadOp.getLoc(),
                op->getResult(0),
                newIndex);
            rewriter.replaceOp(user, newLoad);
        }
        // If it's used in memref.store
        else if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
            auto newIndex = calculateFlattenedIndex(storeOp);
            auto newStore = rewriter.create<memref::StoreOp>(
                storeOp.getLoc(),
                storeOp.getValueToStore(),
                storeOp.getMemRef(),
                newIndex);
            rewriter.replaceOp(user, newStore);
        }
    }
}

template<typename OpT>
struct FlattenProcessRegion : public OpRewritePattern<OpT> {
    TypeConverter converter;
    FlattenProcessRegion(MLIRContext* context, TypeConverter &converter)
            : OpRewritePattern<OpT>(context),
              converter(converter){};

    LogicalResult
    matchAndRewrite(OpT op, PatternRewriter &rewriter) const override
    {
        auto funcTy = op.getFunctionType();
        auto newFuncTy = dyn_cast<FunctionType>(converter.convertType(funcTy));

        auto newOp =
            rewriter.create<OpT>(op.getLoc(), op.getSymNameAttr(), newFuncTy);
        Block* processBlock = &newOp.getBody().front();
        IRMapping mapper;
        for (auto [oldArg, newArg] : llvm::zip(
                 op.getBody().getArguments(),
                 newOp.getBody().getArguments()))
            mapper.map(oldArg, newArg);
        rewriter.setInsertionPointToEnd(processBlock);
        for (auto &opi : op.getBody().getOps()) rewriter.clone(opi, mapper);

        rewriter.replaceOp(op, newOp);
        return success();
    }
};

struct FlattenPull : public OpRewritePattern<PullOp> {
    TypeConverter converter;
    FlattenPull(MLIRContext* context, TypeConverter &converter)
            : OpRewritePattern<PullOp>(context),
              converter(converter){};

    LogicalResult
    matchAndRewrite(PullOp op, PatternRewriter &rewriter) const override
    {
        auto memrefTy = dyn_cast<MemRefType>(op.getResult().getType());
        auto newPull = rewriter.create<PullOp>(
            op.getLoc(),
            converter.convertType(memrefTy),
            op.getChan());
        op.getResult().replaceAllUsesWith(newPull.getResult());
        rewriter.replaceOp(op, newPull);

        flattenUsers(newPull.getOperation(), memrefTy, rewriter);
        return success();
    }
};

template<typename OpT>
struct FlattenAlloc : public OpRewritePattern<OpT> {
    TypeConverter converter;
    FlattenAlloc(MLIRContext* context, TypeConverter &converter)
            : OpRewritePattern<OpT>(context),
              converter(converter){};

    LogicalResult
    matchAndRewrite(OpT op, PatternRewriter &rewriter) const override
    {
        auto memrefTy = op.getType();
        auto newMemrefTy =
            dyn_cast<MemRefType>(converter.convertType(memrefTy));
        auto newAlloc = rewriter.create<OpT>(
            op.getLoc(),
            newMemrefTy,
            op.getAlignmentAttr());
        op.getResult().replaceAllUsesWith(newAlloc.getResult());
        rewriter.replaceOp(op, newAlloc);

        flattenUsers(newAlloc.getOperation(), memrefTy, rewriter);
        return success();
    }
};

struct FlattenChannel : public OpRewritePattern<ChannelOp> {
    TypeConverter converter;
    FlattenChannel(MLIRContext* context, TypeConverter &converter)
            : OpRewritePattern<ChannelOp>(context),
              converter(converter){};

    LogicalResult
    matchAndRewrite(ChannelOp op, PatternRewriter &rewriter) const override
    {
        auto memrefTy = dyn_cast<MemRefType>(op.getEncapsulatedType());
        auto newChannel = rewriter.create<ChannelOp>(
            op.getLoc(),
            converter.convertType(memrefTy),
            *op.getBufferSize());
        op.getResult(0).replaceAllUsesWith(newChannel.getResult(0));
        op.getResult(1).replaceAllUsesWith(newChannel.getResult(1));
        rewriter.replaceOp(op, newChannel);
        return success();
    }
};
} // namespace

void mlir::dfg::populateFlattenMemrefPatterns(
    RewritePatternSet &patterns,
    TypeConverter &converter)
{
    patterns.add<FlattenProcessRegion<ProcessOp>>(
        patterns.getContext(),
        converter);
    patterns.add<FlattenPull>(patterns.getContext(), converter);
    patterns.add<FlattenAlloc<memref::AllocOp>>(
        patterns.getContext(),
        converter);
    patterns.add<FlattenAlloc<memref::AllocaOp>>(
        patterns.getContext(),
        converter);
    patterns.add<FlattenProcessRegion<RegionOp>>(
        patterns.getContext(),
        converter);
    patterns.add<FlattenChannel>(patterns.getContext(), converter);
}

namespace {
struct DfgFlattenMemrefPass
        : public dfg::impl::DfgFlattenMemrefBase<DfgFlattenMemrefPass> {
    void runOnOperation() override;
};
} // namespace

void DfgFlattenMemrefPass::runOnOperation()
{
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    converter.addConversion([](MemRefType memrefTy) {
        auto shape = memrefTy.getShape();
        if (shape.size() == 1) return memrefTy;
        auto elemTy = memrefTy.getElementType();
        auto newShape = std::accumulate(
            shape.begin(),
            shape.end(),
            1,
            std::multiplies<int64_t>{});
        return MemRefType::get(newShape, elemTy);
    });
    converter.addConversion([&](InputType inTy) {
        auto channelTy = inTy.getElementType();
        return InputType::get(&getContext(), converter.convertType(channelTy));
    });
    converter.addConversion([&](OutputType outTy) {
        auto channelTy = outTy.getElementType();
        return OutputType::get(&getContext(), converter.convertType(channelTy));
    });
    converter.addConversion([&](FunctionType funcTy) {
        SmallVector<Type> inTypes, outTypes;
        for (auto inTy : funcTy.getInputs())
            inTypes.push_back(converter.convertType(inTy));
        for (auto outTy : funcTy.getResults())
            outTypes.push_back(converter.convertType(outTy));
        return FunctionType::get(&getContext(), inTypes, outTypes);
    });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateFlattenMemrefPatterns(patterns, converter);

    target.addDynamicallyLegalDialect<DfgDialect, memref::MemRefDialect>(
        [&converter](Operation* op) { return converter.isLegal(op); });
    target.addDynamicallyLegalOp<ProcessOp>([&converter](ProcessOp process) {
        auto funcTy = process.getFunctionType();
        return converter.isLegal(funcTy);
    });
    target.addDynamicallyLegalOp<RegionOp>([&converter](RegionOp region) {
        auto funcTy = region.getFunctionType();
        return converter.isLegal(funcTy);
    });
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::dfg::createDfgFlattenMemrefPass()
{
    return std::make_unique<DfgFlattenMemrefPass>();
}
