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
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include <optional>
#include <string>

namespace mlir {
#define GEN_PASS_DEF_CONVERTDFGTOVITIS
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;

namespace {

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
        auto funcTy = cast<FunctionType>(
            converter->convertType(adaptor.getFunctionType()));

        loopedChannels.clear();
        iterArgValues.clear();
        auto funcOp = rewriter.create<vitis::FuncOp>(
            loc,
            op.getSymName(),
            funcTy,
            ArrayAttr{},
            ArrayAttr{});
        auto funcLoc = funcOp.getLoc();
        Block* funcEntryBlock = rewriter.createBlock(&funcOp.getBody());
        IRMapping mapper;
        for (auto type : funcTy.getInputs())
            funcEntryBlock->addArgument(type, funcLoc);

        rewriter.setInsertionPointToStart(funcEntryBlock);
        for (auto [arg, newArg] :
             llvm::zip(op.getBody().getArguments(), funcOp.getArguments())) {
            auto casted = converter->materializeSourceConversion(
                rewriter,
                funcLoc,
                arg.getType(),
                newArg);
            mapper.map(arg, casted);
        }
        for (auto &op : op.getBody().getOps()) {
            if (auto loopOp = dyn_cast<LoopOp>(op)) {
                for (auto &loopBodyOp : loopOp.getBody().getOps())
                    rewriter.clone(loopBodyOp, mapper);
                iterArgValues.append(
                    loopOp.getIterArgs().begin(),
                    loopOp.getIterArgs().end());
            } else {
                rewriter.clone(op, mapper);
            }
        }

        rewriter.replaceOp(op, funcOp);
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

        if (!isa<MemRefType>(pulledValue.getType())) {
            auto readOp = rewriter.create<vitis::StreamReadOp>(
                loc,
                dyn_cast<vitis::StreamType>(vitisStream.getType())
                    .getStreamType(),
                vitisStream);
            rewriter.replaceOp(op, readOp.getResult());
            return success();
        } else {
            // Create an array to store the pulled data
            auto memrefTy = cast<MemRefType>(pulledValue.getType());
            auto size = memrefTy.getShape().front();
            auto elemTy = memrefTy.getElementType();
            auto array = rewriter.create<vitis::VariableOp>(
                loc,
                vitis::ArrayType::get(rewriter.getContext(), size, elemTy),
                Value{});
            // Create a loop to repeatedly read from the channel
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
            rewriter.create<vitis::ArrayWriteOp>(
                curLoc,
                streamReadOp.getResult(),
                array.getResult(),
                forOp.getInductionVar());

            rewriter.replaceOp(op, array.getResult());
            return success();
        }

        return rewriter.notifyMatchFailure(loc, "Unable to convert pull.");
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
        auto pushChan = op.getChan();
        auto pushedValue = op.getInp();

        auto vitisStream =
            pushChan.getDefiningOp<UnrealizedConversionCastOp>()->getOperand(0);

        if (!isa<MemRefType>(pushedValue.getType())) {
            auto writeOp = rewriter.create<vitis::StreamWriteOp>(
                loc,
                pushedValue,
                vitisStream);
            rewriter.replaceOp(op, writeOp);
            return success();
        } else {
            // The array was casted back to memref using unrealized casts
            auto array = pushedValue.getDefiningOp()->getOperand(0);
            // Create an array to store the pulled data
            auto memrefTy = cast<MemRefType>(pushedValue.getType());
            auto size = memrefTy.getShape().front();
            // Create a loop to repeatedly read from the channel
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
            auto forOp = rewriter.create<vitis::ForOp>(
                loc,
                cstLb.getResult(),
                cstUb.getResult(),
                cstStep.getResult());
            rewriter.setInsertionPointToEnd(&forOp.getBody().front());
            auto curLoc = forOp.getLoc();
            auto arrayReadOp = rewriter.create<vitis::ArrayReadOp>(
                curLoc,
                array,
                forOp.getInductionVar());
            rewriter.create<vitis::StreamWriteOp>(
                curLoc,
                arrayReadOp.getResult(),
                vitisStream);

            rewriter.replaceOp(op, forOp);
            return success();
        }

        return rewriter.notifyMatchFailure(loc, "Unable to convert pull.");
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
    patterns.add<ConvertPullToStreamRead>(typeConverter, patterns.getContext());
    patterns.add<ConvertPushToStreamWrite>(
        typeConverter,
        patterns.getContext());
    // patterns.add<ConvertYieldToUpdates>(typeConverter,
    // patterns.getContext());
    patterns.add<EraseRegion>(typeConverter, patterns.getContext());
}

namespace {
struct ConvertDfgToVitisPass
        : public impl::ConvertDfgToVitisBase<ConvertDfgToVitisPass> {
    void runOnOperation() final;
};
} // namespace

namespace {
struct DfgTypeConverter : public TypeConverter {
public:
    DfgTypeConverter(MLIRContext* context) : context(context)
    {
        addConversion([](Type type) { return type; });
        addConversion([this](InputType type) -> Type {
            auto channelTy = type.getElementType();
            Type elemTy;
            if (auto memrefTy = dyn_cast<MemRefType>(channelTy))
                elemTy = memrefTy.getElementType();
            else
                elemTy = channelTy;
            return vitis::StreamType::get(this->context, elemTy);
        });
        addConversion([this](OutputType type) -> Type {
            auto channelTy = type.getElementType();
            Type elemTy;
            if (auto memrefTy = dyn_cast<MemRefType>(channelTy))
                elemTy = memrefTy.getElementType();
            else
                elemTy = channelTy;
            return vitis::StreamType::get(this->context, elemTy);
        });
        addConversion([this](FunctionType type) -> Type {
            SmallVector<Type> args;
            for (auto inTy : type.getInputs())
                args.push_back(this->convertType(inTy));
            for (auto outTy : type.getResults())
                args.push_back(this->convertType(outTy));
            return FunctionType::get(this->context, args, TypeRange{});
        });
        addSourceMaterialization(materializeAsUnrealizedCast);
        addArgumentMaterialization(materializeAsUnrealizedCast);
        addTargetMaterialization(materializeAsUnrealizedCast);
    }

private:
    MLIRContext* context;
    static Value materializeAsUnrealizedCast(
        OpBuilder &builder,
        Type resultType,
        ValueRange inputs,
        Location loc)
    {
        if (inputs.size() != 1) return Value();
        return builder
            .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
            .getResult(0);
    }
};
} // namespace

void ConvertDfgToVitisPass::runOnOperation()
{
    auto &context = getContext();

    DfgTypeConverter converter(&context);
    ConversionTarget target(context);
    RewritePatternSet patterns(&context);

    populateDfgToVitisConversionPatterns(converter, patterns);

    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addLegalDialect<vitis::VitisDialect>();
    // target.addIllegalDialect<DfgDialect>();
    target.addLegalDialect<DfgDialect>();
    target.addIllegalOp<ProcessOp, RegionOp, PullOp, PushOp>();
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

void mlir::dfg::addConvertToVitisPasses(OpPassManager &pm)
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

void mlir::dfg::registerConvertToVitisPipelines()
{
    PassPipelineRegistration<>(
        "convert-to-vitis",
        "Lower everything to vitis dialect",
        [](OpPassManager &pm) { addConvertToVitisPasses(pm); });
}

void mlir::dfg::addPrepareForVivadoPasses(OpPassManager &pm)
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

void mlir::dfg::registerPrepareForVivadoPipelines()
{
    PassPipelineRegistration<>(
        "prepare-for-vivado",
        "Lower everything to vitis dialect",
        [](OpPassManager &pm) { addPrepareForVivadoPasses(pm); });
}
