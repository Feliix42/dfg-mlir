/// Implementation of DfgToVitis pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/DfgToVitis/DfgToVitis.h"

#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "dfg-mlir/Dialect/vitis/IR/Dialect.h"
#include "dfg-mlir/Dialect/vitis/IR/Ops.h"
#include "dfg-mlir/Dialect/vitis/IR/Types.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

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
            : OpConversionPattern<ProcessOp>(typeConverter, context) {};

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
            : OpConversionPattern<LoopOp>(typeConverter, context) {};

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
            : OpConversionPattern<PullOp>(typeConverter, context) {};

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

        auto streamReadOp = rewriter.create<vitis::StreamReadOp>(
            loc,
            dyn_cast<vitis::StreamType>(vitisStream.getType()).getStreamType(),
            vitisStream);

        for (auto user : userOps) {
            if (isa<PushOp>(user))
                user->replaceUsesOfWith(pulledValue, streamReadOp.getResult());
            else {
                auto dataOp = rewriter.create<vitis::StreamGetDataOp>(
                    loc,
                    pulledValue.getType(),
                    streamReadOp.getResult());
                user->replaceUsesOfWith(pulledValue, dataOp.getResult());
            }
        }

        if (isInSmallVector<Value>(pullChan, loopedChannels)) {
            auto idx = getVectorIdx<Value>(pullChan, loopedChannels).value();
            auto lastOp = rewriter.create<vitis::StreamGetLastOp>(
                loc,
                streamReadOp.getResult());
            loopedChannels[idx] = lastOp.getResult();

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
            : OpConversionPattern<PushOp>(typeConverter, context) {};

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
        else {
            Type varTy;
            if (pushedValue.getType().isUnsignedInteger()) {
                varTy = vitis::APAxiUType::get(
                    ctx,
                    pushedValue.getType().getIntOrFloatBitWidth(),
                    0,
                    0,
                    0,
                    true);
            } else {
                varTy = vitis::APAxiSType::get(
                    ctx,
                    pushedValue.getType().getIntOrFloatBitWidth(),
                    0,
                    0,
                    0,
                    true);
            }
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
            : OpConversionPattern<YieldOp>(typeConverter, context) {};

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
            : OpConversionPattern<RegionOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        RegionOp op,
        RegionOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.eraseOp(op);
        return success();
    }
};

struct ConvertArithConstant : OpConversionPattern<arith::ConstantOp> {

    using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

    ConvertArithConstant(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<arith::ConstantOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        arith::ConstantOp op,
        arith::ConstantOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto constantAttr = dyn_cast<IntegerAttr>(op.getValue());
        if (!constantAttr)
            return rewriter.notifyMatchFailure(
                loc,
                "Only integer constants are supported for hardware.");

        auto constantOp = rewriter.create<vitis::ConstantOp>(loc, constantAttr);
        auto variableOp = rewriter.replaceOpWithNewOp<vitis::VariableOp>(
            op,
            op.getType(),
            constantOp.getResult());
        op.getResult().replaceAllUsesWith(variableOp.getResult());

        return success();
    }
};

template<typename OpFrom, typename OpTo>
struct ConvertArithBinaryOp : OpConversionPattern<OpFrom> {

    using OpConversionPattern<OpFrom>::OpConversionPattern;

    ConvertArithBinaryOp(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<OpFrom>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        OpFrom op,
        OpFrom::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<OpTo>(op, op.getLhs(), op.getRhs());
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
    patterns.add<ConvertArithConstant>(typeConverter, patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::AddIOp, vitis::ArithAddOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::MulIOp, vitis::ArithMulOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::AndIOp, vitis::ArithAndOp>>(
        typeConverter,
        patterns.getContext());
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
        auto elemTy = type.getElementType();
        if (!isa<IntegerType>(elemTy))
            return Type();
        else {
            auto bitwidth = elemTy.getIntOrFloatBitWidth();
            if (elemTy.isUnsignedInteger())
                return vitis::StreamType::get(
                    &getContext(),
                    vitis::APAxiUType::get(
                        &getContext(),
                        bitwidth,
                        0,
                        0,
                        0,
                        true));
            else
                return vitis::StreamType::get(
                    &getContext(),
                    vitis::APAxiSType::get(
                        &getContext(),
                        bitwidth,
                        0,
                        0,
                        0,
                        true));
        }
    });
    converter.addConversion([&](OutputType type) -> Type {
        auto elemTy = type.getElementType();
        if (!isa<IntegerType>(elemTy))
            return Type();
        else {
            auto bitwidth = elemTy.getIntOrFloatBitWidth();
            if (elemTy.isUnsignedInteger())
                return vitis::StreamType::get(
                    &getContext(),
                    vitis::APAxiUType::get(
                        &getContext(),
                        bitwidth,
                        0,
                        0,
                        0,
                        true));
            else
                return vitis::StreamType::get(
                    &getContext(),
                    vitis::APAxiSType::get(
                        &getContext(),
                        bitwidth,
                        0,
                        0,
                        0,
                        true));
        }
    });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateDfgToVitisConversionPatterns(converter, patterns);

    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addLegalDialect<vitis::VitisDialect>();
    target.addIllegalDialect<arith::ArithDialect, DfgDialect>();

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
