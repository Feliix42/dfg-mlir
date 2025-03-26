/// Implementation of DfgToVitis pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/DfgToVitis/DfgToVitis.h"

#include "dfg-mlir/Conversion/Passes.h"
#include "dfg-mlir/Conversion/ScfToVitis/ScfToVitis.h"
#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h"
#include "dfg-mlir/Dialect/vitis/Enums.h"
#include "dfg-mlir/Dialect/vitis/IR/Dialect.h"
#include "dfg-mlir/Dialect/vitis/IR/Ops.h"
#include "dfg-mlir/Dialect/vitis/IR/Types.h"
#include "dfg-mlir/Dialect/vitis/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/APInt.h"

#include <cstddef>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Process.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_CONVERTDFGTOVITIS
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;

namespace {

// SmallVector<TypedValue<OutputType>> loopedChannels;
SmallVector<Value> loopedChannels, iterArgValues;
Value lastSignal, lastSignalPlaceholder;

struct ConvertProcessToFunc : OpConversionPattern<ProcessOp> {
    using OpConversionPattern<ProcessOp>::OpConversionPattern;

    ConvertProcessToFunc(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<ProcessOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        ProcessOp op,
        ProcessOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto converter = getTypeConverter();
        auto loc = op->getLoc();

        SmallVector<Type> types;
        for (auto inTy : adaptor.getFunctionType().getInputs())
            types.push_back(converter->convertType(inTy));
        for (auto outTy : adaptor.getFunctionType().getResults())
            types.push_back(converter->convertType(outTy));

        loopedChannels.clear();
        iterArgValues.clear();
        auto funcOp = rewriter.create<vitis::FuncOp>(
            loc,
            op.getSymName(),
            rewriter.getFunctionType(types, {}),
            ArrayAttr{},
            ArrayAttr{});
        auto funcLoc = funcOp.getLoc();
        Block* funcEntryBlock = rewriter.createBlock(&funcOp.getBody());
        IRMapping mapper;
        for (auto type : types) funcEntryBlock->addArgument(type, funcLoc);

        rewriter.setInsertionPointToStart(funcEntryBlock);
        for (auto [arg, newArg] :
             llvm::zip(op.getBody().getArguments(), funcOp.getArguments())) {
            auto castOp = rewriter.create<UnrealizedConversionCastOp>(
                funcLoc,
                TypeRange{arg.getType()},
                newArg);
            mapper.map(arg, castOp->getResult(0));
        }
        for (auto &op : op.getBody().getOps()) rewriter.clone(op, mapper);
        rewriter.create<vitis::ReturnOp>(funcOp->getLoc(), TypeRange{});

        rewriter.eraseOp(op);
        return success();
    }
};

struct ConvertLoopToWhile : OpConversionPattern<LoopOp> {
    using OpConversionPattern<LoopOp>::OpConversionPattern;

    ConvertLoopToWhile(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<LoopOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        LoopOp op,
        LoopOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        loopedChannels.append(op.getInChans().begin(), op.getInChans().end());
        iterArgValues.append(op.getIterArgs().begin(), op.getIterArgs().end());

        auto whileTrueOp = rewriter.create<vitis::WhileTrueOp>(loc);
        Block &whileBody = whileTrueOp.getBody().emplaceBlock();
        rewriter.setInsertionPointToStart(&whileBody);
        IRMapping mapper;
        for (auto &opi : op.getBody().getOps()) rewriter.clone(opi, mapper);
        lastSignalPlaceholder =
            rewriter
                .create<vitis::ConstantOp>(
                    loc,
                    rewriter.getIntegerAttr(rewriter.getI1Type(), 0))
                .getResult();
        rewriter.create<vitis::IfBreakOp>(loc, lastSignalPlaceholder);

        rewriter.eraseOp(op);
        return success();
    }
};

struct ConvertPullToStreamRead : OpConversionPattern<PullOp> {
    using OpConversionPattern<PullOp>::OpConversionPattern;

    ConvertPullToStreamRead(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<PullOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        PullOp op,
        PullOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto pullChan = op.getChan();
        auto pulledValue = op.getResult();

        auto vitisStream =
            pullChan.getDefiningOp<UnrealizedConversionCastOp>()->getOperand(0);
        auto userOps = pulledValue.getUsers();

        vitis::StreamReadOp readOp;
        if (!isa<MemRefType>(pulledValue.getType()))
            readOp = rewriter.create<vitis::StreamReadOp>(
                loc,
                dyn_cast<vitis::StreamType>(vitisStream.getType())
                    .getStreamType(),
                vitisStream);

        // If the pulled value is a memref, this is the last signal
        Value lastVar;
        for (auto user : userOps) {
            if (isa<PushOp>(user)) {
                user->replaceUsesOfWith(pulledValue, readOp.getResult());
            } else {
                auto pulledType = pulledValue.getType();
                // If the pulled value is a memref, create a for-loop to read
                // the data into an array
                if (auto memrefTy = dyn_cast<MemRefType>(pulledType)) {
                    auto elemTy = memrefTy.getElementType();
                    auto size = memrefTy.getShape().front();
                    auto array = rewriter.create<vitis::VariableOp>(
                        loc,
                        vitis::ArrayType::get(
                            rewriter.getContext(),
                            size,
                            elemTy),
                        Value{});
                    user->replaceUsesOfWith(pulledValue, array.getResult());
                    auto last = rewriter.create<vitis::VariableOp>(
                        loc,
                        rewriter.getI1Type(),
                        Value{});
                    lastVar = last.getResult();

                    // Deal with memref.load/store ops
                    rewriter.setInsertionPoint(user);
                    // If it's used in memref.load
                    if (auto loadOp = dyn_cast<memref::LoadOp>(user)) {
                        auto arrayRead = rewriter.create<vitis::ArrayReadOp>(
                            loadOp.getLoc(),
                            memrefTy.getElementType(),
                            array.getResult(),
                            loadOp.getIndices().front());
                        rewriter.replaceOp(user, arrayRead);
                    }
                    // If it's used in memref.store
                    else if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
                        auto arrayWrite = rewriter.create<vitis::ArrayWriteOp>(
                            storeOp.getLoc(),
                            storeOp.getValueToStore(),
                            array.getResult(),
                            storeOp.getIndices().front());
                        rewriter.replaceOp(user, arrayWrite);
                    }
                    rewriter.setInsertionPoint(op);

                    // Create the index constants outside of the loop
                    rewriter.setInsertionPoint(
                        op->getParentOfType<vitis::WhileTrueOp>());
                    auto cstLb = rewriter.create<arith::ConstantOp>(
                        loc,
                        rewriter.getIndexType(),
                        rewriter.getIndexAttr(0));
                    auto cstUb = rewriter.create<arith::ConstantOp>(
                        loc,
                        rewriter.getIndexType(),
                        rewriter.getIndexAttr(size));
                    auto cstStep = rewriter.create<arith::ConstantOp>(
                        loc,
                        rewriter.getIndexType(),
                        rewriter.getIndexAttr(1));
                    rewriter.setInsertionPoint(op);
                    // Create the loop
                    auto forOp = rewriter.create<vitis::ForOp>(
                        loc,
                        cstLb.getResult(),
                        cstUb.getResult(),
                        cstStep.getResult());
                    rewriter.setInsertionPointToEnd(&forOp.getBody().front());
                    auto curLoc = forOp.getLoc();
                    auto streamReadOp = rewriter.create<vitis::StreamReadOp>(
                        curLoc,
                        dyn_cast<vitis::StreamType>(vitisStream.getType())
                            .getStreamType(),
                        vitisStream);
                    auto streamDataOp = rewriter.create<vitis::StreamGetDataOp>(
                        curLoc,
                        dyn_cast<vitis::APAxisType>(
                            streamReadOp.getResult().getType())
                            .getElemType(),
                        streamReadOp.getResult());
                    rewriter.create<vitis::ArrayWriteOp>(
                        curLoc,
                        streamDataOp.getResult(),
                        array.getResult(),
                        forOp.getInductionVar());
                    auto streamLastOp = rewriter.create<vitis::StreamGetLastOp>(
                        curLoc,
                        streamReadOp.getResult());
                    rewriter.create<vitis::UpdateOp>(
                        curLoc,
                        last.getResult(),
                        streamLastOp.getResult());
                    rewriter.setInsertionPoint(op);
                }
                // If it's a scalar value, create a stream get data op
                else {
                    auto dataOp = rewriter.create<vitis::StreamGetDataOp>(
                        loc,
                        pulledType,
                        readOp.getResult());
                    user->replaceUsesOfWith(pulledValue, dataOp.getResult());
                }
            }
        }

        if (isInSmallVector<Value>(pullChan, loopedChannels)) {
            size_t idx = getVectorIdx<Value>(pullChan, loopedChannels).value();
            if (isa<MemRefType>(pulledValue.getType())) {
                loopedChannels[idx] = lastVar;
            } else {
                auto lastOp = rewriter.create<vitis::StreamGetLastOp>(
                    loc,
                    readOp.getResult());
                loopedChannels[idx] = lastOp.getResult();
            }

            // If this is the last pull, calculate the last signal
            if (idx == loopedChannels.size() - 1) {
                for (size_t i = 0; i < loopedChannels.size(); i++) {
                    if (i == 0)
                        lastSignal = loopedChannels[i];
                    else {
                        auto andOp = rewriter.create<arith::AndIOp>(
                            loc,
                            lastSignal,
                            loopedChannels[i]);
                        lastSignal = andOp.getResult();
                    }
                }
                lastSignalPlaceholder.replaceAllUsesWith(lastSignal);
                rewriter.eraseOp(lastSignalPlaceholder.getDefiningOp());
            }
        }

        rewriter.eraseOp(op);
        return success();
    }
};

struct ConvertPushToStreamWrite : OpConversionPattern<PushOp> {
    using OpConversionPattern<PushOp>::OpConversionPattern;

    ConvertPushToStreamWrite(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<PushOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        PushOp op,
        PushOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();
        auto pushChan = op.getChan();
        auto pushedValue = op.getInp();

        auto vitisStream =
            pushChan.getDefiningOp<UnrealizedConversionCastOp>()->getOperand(0);

        if (pushedValue.getType().getDialect().getNamespace() == "vitis")
            rewriter.create<vitis::StreamWriteOp>(
                loc,
                pushedValue,
                vitisStream);
        else if (auto memrefTy = dyn_cast<MemRefType>(pushedValue.getType())) {
            auto pushedType = memrefTy.getElementType();
            auto size = memrefTy.getShape().front();
            rewriter.setInsertionPoint(
                op->getParentOfType<vitis::WhileTrueOp>());
            auto cstFalse = rewriter.create<arith::ConstantOp>(
                loc,
                rewriter.getI1Type(),
                rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
            auto cstLb = rewriter.create<arith::ConstantOp>(
                loc,
                rewriter.getIndexType(),
                rewriter.getIndexAttr(0));
            auto cstUb = rewriter.create<arith::ConstantOp>(
                loc,
                rewriter.getIndexType(),
                rewriter.getIndexAttr(size));
            auto cstStep = rewriter.create<arith::ConstantOp>(
                loc,
                rewriter.getIndexType(),
                rewriter.getIndexAttr(1));
            auto cstMaxIdx = rewriter.create<arith::ConstantOp>(
                loc,
                rewriter.getIndexType(),
                rewriter.getIndexAttr(size - 1));
            rewriter.setInsertionPoint(op);
            // Create the loop
            auto forOp = rewriter.create<vitis::ForOp>(
                loc,
                cstLb.getResult(),
                cstUb.getResult(),
                cstStep.getResult());
            rewriter.setInsertionPointToEnd(&forOp.getBody().front());

            auto curLoc = forOp.getLoc();
            Type varTy = vitis::APAxisType::get(ctx, pushedType, 0, 0, 0, true);
            auto variableOp =
                rewriter.create<vitis::VariableOp>(curLoc, varTy, Value());
            auto data = rewriter.create<vitis::ArrayReadOp>(
                curLoc,
                pushedValue.getDefiningOp()->getOperand(0),
                forOp.getInductionVar());
            rewriter.create<vitis::StreamSetDataOp>(
                curLoc,
                data.getResult(),
                variableOp.getResult());
            auto cmpOp = rewriter.create<vitis::ArithCmpOp>(
                curLoc,
                vitis::CmpPredicateAttr::get(
                    rewriter.getContext(),
                    vitis::CmpPredicate::eq),
                forOp.getInductionVar(),
                cstMaxIdx.getResult());
            auto selectOp = rewriter.create<vitis::ArithSelectOp>(
                curLoc,
                cmpOp.getResult(),
                lastSignal,
                cstFalse.getResult());
            rewriter.create<vitis::StreamSetLastOp>(
                loc,
                selectOp.getResult(),
                variableOp.getResult());
            rewriter.create<vitis::StreamWriteOp>(
                loc,
                variableOp.getResult(),
                vitisStream);
        } else {
            Type varTy = vitis::APAxisType::get(
                ctx,
                pushedValue.getType(),
                0,
                0,
                0,
                true);
            auto variableOp =
                rewriter.create<vitis::VariableOp>(loc, varTy, Value());

            rewriter.create<vitis::StreamSetDataOp>(
                loc,
                pushedValue,
                variableOp.getResult());
            rewriter.create<vitis::StreamSetLastOp>(
                loc,
                lastSignal,
                variableOp.getResult());
            rewriter.create<vitis::StreamWriteOp>(
                loc,
                variableOp.getResult(),
                vitisStream);
        }

        rewriter.eraseOp(op);
        return success();
    }
};

struct ConvertYieldToUpdates : OpConversionPattern<YieldOp> {
    using OpConversionPattern<YieldOp>::OpConversionPattern;

    ConvertYieldToUpdates(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<YieldOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        YieldOp op,
        YieldOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();

        for (auto [iterArg, newValue] :
             llvm::zip(iterArgValues, op.getOperands())) {
            rewriter.create<vitis::UpdateOp>(loc, iterArg, newValue);
        }

        rewriter.eraseOp(op);
        return success();
    }
};

struct EraseRegion : OpConversionPattern<RegionOp> {
    using OpConversionPattern<RegionOp>::OpConversionPattern;

    EraseRegion(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<RegionOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        RegionOp op,
        RegionOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.eraseOp(op);
        return success();
    }
};
} // namespace

void mlir::populateDfgToVitisConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<ConvertProcessToFunc>(typeConverter, patterns.getContext());
    patterns.add<ConvertLoopToWhile>(typeConverter, patterns.getContext());
    patterns.add<ConvertPullToStreamRead>(typeConverter, patterns.getContext());
    patterns.add<ConvertPushToStreamWrite>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertYieldToUpdates>(typeConverter, patterns.getContext());
    patterns.add<EraseRegion>(typeConverter, patterns.getContext());
}

namespace {
struct ConvertDfgToVitisPass
        : public impl::ConvertDfgToVitisBase<ConvertDfgToVitisPass> {
    void runOnOperation() final;
};
} // namespace

void ConvertDfgToVitisPass::runOnOperation()
{
    TypeConverter converter;

    converter.addConversion([&](Type type) { return type; });
    converter.addConversion([&](InputType type) -> Type {
        auto channelTy = type.getElementType();
        Type elemTy;
        if (auto memrefTy = dyn_cast<MemRefType>(channelTy))
            elemTy = memrefTy.getElementType();
        else
            elemTy = channelTy;
        return vitis::StreamType::get(
            &getContext(),
            vitis::APAxisType::get(&getContext(), elemTy, 0, 0, 0, true));
    });
    converter.addConversion([&](OutputType type) -> Type {
        auto channelTy = type.getElementType();
        Type elemTy;
        if (auto memrefTy = dyn_cast<MemRefType>(channelTy))
            elemTy = memrefTy.getElementType();
        else
            elemTy = channelTy;
        return vitis::StreamType::get(
            &getContext(),
            vitis::APAxisType::get(&getContext(), elemTy, 0, 0, 0, true));
    });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateDfgToVitisConversionPatterns(converter, patterns);

    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addLegalDialect<vitis::VitisDialect>();
    target.addIllegalDialect<DfgDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertDfgToVitisPass()
{
    return std::make_unique<ConvertDfgToVitisPass>();
}

void mlir::addConvertToVitisPasses(OpPassManager &pm)
{
    pm.addPass(dfg::createDfgOperatorToProcessPass());
    pm.addPass(dfg::createDfgInlineRegionPass());
    pm.addPass(dfg::createDfgLowerInsideToLinalgPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(bufferization::createOneShotBufferizePass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createConvertLinalgToLoopsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(dfg::createDfgFlattenMemrefPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createConvertMemrefToVitisPass());
    pm.addPass(createConvertDfgToVitisPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createConvertArithIndexToVitisPass());
    pm.addPass(createConvertMathToVitisPass());
    pm.addPass(createConvertScfToVitisPass());
    pm.addPass(vitis::createVitisMergeCastChainPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
}

void mlir::registerConvertToVitisPipelines()
{
    PassPipelineRegistration<>(
        "convert-to-vitis",
        "Lower everything to vitis dialect",
        [](OpPassManager &pm) { addConvertToVitisPasses(pm); });
}

void mlir::addPrepareForVivadoPasses(OpPassManager &pm)
{
    pm.addPass(dfg::createDfgOperatorToProcessPass());
    pm.addPass(dfg::createDfgInlineRegionPass());
    pm.addPass(dfg::createDfgLowerInsideToLinalgPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(bufferization::createOneShotBufferizePass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createConvertLinalgToLoopsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(dfg::createDfgFlattenMemrefPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
}

void mlir::registerPrepareForVivadoPipelines()
{
    PassPipelineRegistration<>(
        "prepare-for-vivado",
        "Lower everything to vitis dialect",
        [](OpPassManager &pm) { addPrepareForVivadoPasses(pm); });
}
