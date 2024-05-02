/// Implementation of the DfgToLLVM pass.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Conversion/DfgToLLVM/DfgToLLVM.h"

#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <iostream>

namespace mlir {
#define GEN_PASS_DEF_CONVERTDFGTOLLVM
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;

Value mainRegion;

Type wrapperType;
Type wrapperPtrType;
Type rtWrapperType;
Type rtWrapperPtrType;
Type channelUntypedType;
Type channelUntypedPtrType;
Type rtChannelUntypedType;
Type rtChannelUntypedPtrType;

// ========================================================
// Lowerings
// ========================================================

namespace {
struct ConvertDfgToLLVMPass
        : public mlir::impl::ConvertDfgToLLVMBase<ConvertDfgToLLVMPass> {
    void runOnOperation() final;
};
} // namespace

struct ChannelOpLowering : public OpConversionPattern<ChannelOp> {
    using OpConversionPattern<ChannelOp>::OpConversionPattern;

    ChannelOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<ChannelOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        ChannelOp channelOp,
        ChannelOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = channelOp.getLoc();

        auto constant = rewriter.create<LLVM::ConstantOp>(
            loc,
            rewriter.getI32Type(),
            rewriter.getI32IntegerAttr(4));

        auto newChannel = rewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{
                typeConverter->convertType(channelOp.getInChan().getType())},
            rewriter.getStringAttr("C_INTERFACE_AddChannel"),
            ValueRange{constant.getResult()});

        rewriter.replaceOp(
            channelOp,
            ValueRange{newChannel.getResult(), newChannel.getResult()});

        return success();
    }
};

struct PushOpLowering : public mlir::OpConversionPattern<PushOp> {
    using OpConversionPattern<PushOp>::OpConversionPattern;

    PushOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<PushOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        PushOp pushOp,
        PushOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(
            pushOp,
            TypeRange{},
            rewriter.getStringAttr("C_INTERFACE_PushBytes"),
            ValueRange{adaptor.getChan(), adaptor.getInp()});
        return success();
    }
};

struct PullOpLowering : public mlir::OpConversionPattern<PullOp> {
    using OpConversionPattern<PullOp>::OpConversionPattern;

    PullOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<PullOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        PullOp pullOp,
        PullOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(
            pullOp,
            TypeRange{pullOp.getOutp()},
            rewriter.getStringAttr("C_INTERFACE_PopBytes"),
            ValueRange{adaptor.getChan()});
        return success();
    }
};

struct OperatorOpLowering_BUT_WE_NEED_TO_CALL_IT_SOMETHING_ELSE_FOR_SOME_REASON
        : public mlir::OpConversionPattern<OperatorOp> {
    using OpConversionPattern<OperatorOp>::OpConversionPattern;

    OperatorOpLowering_BUT_WE_NEED_TO_CALL_IT_SOMETHING_ELSE_FOR_SOME_REASON(
        TypeConverter &typeConverter,
        MLIRContext* context)
            : OpConversionPattern<OperatorOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        OperatorOp operatorOp,
        OperatorOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override

    {

        auto loc = operatorOp.getLoc();

        LLVM::LLVMFunctionType functionType = LLVM::LLVMFunctionType::get(
            rewriter.getContext(),
            LLVM::LLVMVoidType::get(rewriter.getContext()),
            {rtWrapperPtrType, rtWrapperPtrType},
            false);
        auto functionHolder = rewriter.create<LLVM::LLVMFuncOp>(
            loc,
            operatorOp.getSymName(),
            functionType);

        Block* newBlock = new Block();

        newBlock->addArgument(rtWrapperPtrType, adaptor.getBody().getLoc());
        newBlock->addArgument(rtWrapperPtrType, adaptor.getBody().getLoc());

        Value inputWrapper = newBlock->getArgument(0);
        Value outputWrapper = newBlock->getArgument(1);

        rewriter.setInsertionPointToEnd(newBlock);

        std::vector<Value> allValues;

        size_t index = 0;
        for (auto input : adaptor.getFunctionType().getInputs()) {
            auto inputType = typeConverter->convertType(input);
            auto indexValue = rewriter.create<LLVM::ConstantOp>(
                loc,
                rewriter.getI64Type(),
                rewriter.getI64IntegerAttr(index++));
            auto newValue = rewriter.create<LLVM::CallOp>(
                loc,
                TypeRange{inputType},
                rewriter.getStringAttr("RTChannelUntypedPtrsWrapperGet"),
                ValueRange{inputWrapper, indexValue.getResult()});
            allValues.push_back(newValue.getResult());
        }

        index = 0;
        for (auto output : adaptor.getFunctionType().getResults()) {
            auto outputType = typeConverter->convertType(output);
            auto indexValue = rewriter.create<LLVM::ConstantOp>(
                loc,
                rewriter.getI64Type(),
                rewriter.getI64IntegerAttr(index++));
            auto newValue = rewriter.create<LLVM::CallOp>(
                loc,
                TypeRange{outputType},
                rewriter.getStringAttr("RTChannelUntypedPtrsWrapperGet"),
                ValueRange{outputWrapper, indexValue.getResult()});
            allValues.push_back(newValue.getResult());
        }

        rewriter.create<cf::BranchOp>(
            loc,
            allValues,
            &adaptor.getBody().back());
        rewriter.setInsertionPointToEnd(&adaptor.getBody().back());
        newBlock->insertBefore(&adaptor.getBody().back());

        FailureOr<Block*> convertedBlock =
            rewriter.convertRegionTypes(&adaptor.getBody(), *typeConverter);

        if (*convertedBlock == nullptr) return failure();

        rewriter.inlineRegionBefore(
            adaptor.getBody(),
            functionHolder.getBody(),
            functionHolder.end());

        rewriter.setInsertionPointToEnd(&functionHolder.getRegion().back());
        rewriter.create<LLVM::ReturnOp>(loc, TypeRange({}), ValueRange({}));
        rewriter.eraseOp(operatorOp);

        return success();
    }
};

struct
    InstantiateOpLowering_BUT_WE_NEED_TO_CALL_IT_SOMETHING_ELSE_FOR_SOME_REASON
        : public mlir::OpConversionPattern<dfg::InstantiateOp> {
    using OpConversionPattern<dfg::InstantiateOp>::OpConversionPattern;

    InstantiateOpLowering_BUT_WE_NEED_TO_CALL_IT_SOMETHING_ELSE_FOR_SOME_REASON(
        TypeConverter &typeConverter,
        MLIRContext* context)
            : OpConversionPattern<dfg::InstantiateOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        dfg::InstantiateOp instantiateOp,
        dfg::InstantiateOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {

        auto loc = instantiateOp.getLoc();

        auto arraySize = rewriter.create<LLVM::ConstantOp>(
            loc,
            rewriter.getI32Type(),
            rewriter.getI32IntegerAttr(1));

        auto inputWrapper = rewriter.create<LLVM::AllocaOp>(
            loc,
            wrapperPtrType,
            arraySize.getResult(),
            8);

        auto inputSize = rewriter.create<LLVM::ConstantOp>(
            loc,
            rewriter.getI64Type(),
            rewriter.getI64IntegerAttr(adaptor.getInputs().size()));

        rewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{},
            rewriter.getStringAttr("ChannelUntypedPtrsWrapperInit"),
            ValueRange{inputWrapper.getResult(), inputSize});

        size_t index = 0;
        for (auto input : adaptor.getInputs()) {
            rewriter.create<LLVM::CallOp>(
                loc,
                TypeRange{},
                rewriter.getStringAttr("ChannelUntypedPtrsWrapperSet"),
                ValueRange{
                    inputWrapper.getResult(),
                    input,
                    rewriter.create<LLVM::ConstantOp>(
                        loc,
                        rewriter.getI64Type(),
                        rewriter.getI64IntegerAttr(index++))});
        }

        auto outputWrapper = rewriter.create<LLVM::AllocaOp>(
            loc,
            wrapperPtrType,
            arraySize.getResult(),
            8);

        auto outputSize = rewriter.create<LLVM::ConstantOp>(
            loc,
            rewriter.getI64Type(),
            rewriter.getI64IntegerAttr(adaptor.getOutputs().size()));

        rewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{},
            rewriter.getStringAttr("ChannelUntypedPtrsWrapperInit"),
            ValueRange{outputWrapper.getResult(), outputSize});

        index = 0;
        for (auto output : adaptor.getOutputs()) {
            rewriter.create<LLVM::CallOp>(
                loc,
                TypeRange{},
                rewriter.getStringAttr("ChannelUntypedPtrsWrapperSet"),
                ValueRange{
                    outputWrapper.getResult(),
                    output,
                    rewriter.create<LLVM::ConstantOp>(
                        loc,
                        rewriter.getI64Type(),
                        rewriter.getI64IntegerAttr(index++))});
        }

        rewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{LLVM::LLVMPointerType::get(rewriter.getContext())},
            rewriter.getStringAttr("C_INTERFACE_AddKpnProcess"),
            ValueRange{inputWrapper.getResult(), outputWrapper.getResult()});

        rewriter.eraseOp(instantiateOp);

        return success();
    }
};

void ConvertDfgToLLVMPass::runOnOperation()
{
    Operation* op = getOperation();

    TypeConverter highLevelConverter;
    TypeConverter runtimeConverter;

    ConversionPatternRewriter rewriter(&getContext());
    wrapperType = LLVM::LLVMStructType::getNewIdentified(
        &getContext(),
        StringRef("ChannelUntypedPtrsWrapper"),
        {rewriter.getI32Type()});
    wrapperPtrType = LLVM::LLVMPointerType::get(wrapperType);

    rtWrapperType = LLVM::LLVMStructType::getNewIdentified(
        &getContext(),
        StringRef("RTChannelUntypedPtrsWrapper"),
        {rewriter.getI32Type()});
    rtWrapperPtrType = LLVM::LLVMPointerType::get(rtWrapperType);

    channelUntypedType = LLVM::LLVMStructType::getNewIdentified(
        &getContext(),
        StringRef("ChannelUntyped"),
        {rewriter.getI32Type()});
    channelUntypedPtrType = LLVM::LLVMPointerType::get(channelUntypedType);

    rtChannelUntypedType = LLVM::LLVMStructType::getNewIdentified(
        &getContext(),
        StringRef("RTChannelUntyped"),
        {rewriter.getI32Type()});
    rtChannelUntypedPtrType = LLVM::LLVMPointerType::get(rtChannelUntypedType);

    auto regionUntypedPtrType =
        LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getNewIdentified(
            &getContext(),
            StringRef("class.Dppm::RegionUntyped"),
            {rewriter.getI32Type()}));

    auto applicationPtrType =
        LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getNewIdentified(
            &getContext(),
            StringRef("class.Dppm::Application"),
            {rewriter.getI32Type()}));

    auto managerPtrType =
        LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getNewIdentified(
            &getContext(),
            StringRef("class.Dppm::Manager"),
            {rewriter.getI32Type()}));

    highLevelConverter.addConversion([](Type t) { return t; });

    highLevelConverter.addConversion([](InputType t) {
        Type templateType = t.getElementType();
        std::string templateTypeString;
        llvm::raw_string_ostream templateTypeStream(templateTypeString);
        templateType.print(templateTypeStream);
        return channelUntypedPtrType;
    });

    highLevelConverter.addConversion([](OutputType t) {
        Type templateType = t.getElementType();
        std::string templateTypeString;
        llvm::raw_string_ostream templateTypeStream(templateTypeString);
        templateType.print(templateTypeStream);
        return channelUntypedPtrType;
    });

    runtimeConverter.addConversion([](Type t) { return t; });
    runtimeConverter.addConversion([](InputType t) {
        Type templateType = t.getElementType();
        std::string templateTypeString;
        llvm::raw_string_ostream templateTypeStream(templateTypeString);
        templateType.print(templateTypeStream);
        return rtChannelUntypedPtrType;
    });

    runtimeConverter.addConversion([](OutputType t) {
        Type templateType = t.getElementType();
        std::string templateTypeString;
        llvm::raw_string_ostream templateTypeStream(templateTypeString);
        templateType.print(templateTypeStream);
        return rtChannelUntypedPtrType;
    });

    op->walk([&](func::FuncOp funcOp) -> WalkResult {
        if (funcOp.getName() != "main") {
            return WalkResult::advance();
        } else {
            ConversionPatternRewriter localRewriter(funcOp->getContext());
            auto loc = funcOp.getLoc();

            localRewriter.setInsertionPoint(funcOp);
            localRewriter.create<LLVM::LLVMFuncOp>(
                loc,
                "C_INTERFACE_AddChannel",
                LLVM::LLVMFunctionType::get(
                    &getContext(),
                    channelUntypedPtrType,
                    {rewriter.getI32Type()},
                    false));
            localRewriter.create<LLVM::LLVMFuncOp>(
                loc,
                "ChannelUntypedPtrsWrapperSet",
                LLVM::LLVMFunctionType::get(
                    &getContext(),
                    LLVM::LLVMVoidType::get(&getContext()),
                    {wrapperPtrType,
                     channelUntypedPtrType,
                     rewriter.getI64Type()},
                    false));
            localRewriter.create<LLVM::LLVMFuncOp>(
                loc,
                "RTChannelUntypedPtrsWrapperGet",
                LLVM::LLVMFunctionType::get(
                    &getContext(),
                    rtChannelUntypedPtrType,
                    {rtWrapperPtrType, rewriter.getI64Type()},
                    false));
            localRewriter.create<LLVM::LLVMFuncOp>(
                loc,
                "C_INTERFACE_PopBytes",
                LLVM::LLVMFunctionType::get(
                    &getContext(),
                    localRewriter.getI64Type(),
                    {rtChannelUntypedPtrType},
                    false));
            localRewriter.create<LLVM::LLVMFuncOp>(
                loc,
                "C_INTERFACE_PushBytes",
                LLVM::LLVMFunctionType::get(
                    &getContext(),
                    LLVM::LLVMVoidType::get(&getContext()),
                    {rtChannelUntypedPtrType, localRewriter.getI64Type()},
                    false));
            localRewriter.create<LLVM::LLVMFuncOp>(
                loc,
                "C_INTERFACE_GetMainRegion",
                LLVM::LLVMFunctionType::get(
                    &getContext(),
                    regionUntypedPtrType,
                    {applicationPtrType},
                    false));
            localRewriter.create<LLVM::LLVMFuncOp>(
                loc,
                "C_INTERFACE_CreateApplication",
                LLVM::LLVMFunctionType::get(
                    &getContext(),
                    applicationPtrType,
                    {managerPtrType},
                    false));
            localRewriter.create<LLVM::LLVMFuncOp>(
                loc,
                "C_INTERFACE_ManagerGetInstance",
                LLVM::LLVMFunctionType::get(
                    &getContext(),
                    managerPtrType,
                    {},
                    false));
            localRewriter.create<LLVM::LLVMFuncOp>(
                loc,
                "ChannelUntypedPtrsWrapperInit",
                LLVM::LLVMFunctionType::get(
                    &getContext(),
                    LLVM::LLVMVoidType::get(&getContext()),
                    {wrapperPtrType, rewriter.getI64Type()},
                    false));
            localRewriter.create<LLVM::LLVMFuncOp>(
                loc,
                "C_INTERFACE_AddKpnProcess",
                LLVM::LLVMFunctionType::get(
                    &getContext(),
                    LLVM::LLVMPointerType::get(rewriter.getContext()),
                    {wrapperPtrType, wrapperPtrType},
                    false));
            localRewriter.setInsertionPointToStart(&funcOp.getBody().front());

            auto manager = localRewriter.create<LLVM::CallOp>(
                loc,
                TypeRange{managerPtrType},
                localRewriter.getStringAttr("C_INTERFACE_ManagerGetInstance"),
                ValueRange{});

            auto application = localRewriter.create<LLVM::CallOp>(
                loc,
                TypeRange{applicationPtrType},
                localRewriter.getStringAttr("C_INTERFACE_CreateApplication"),
                ValueRange{manager.getResult()});

            mainRegion = localRewriter
                             .create<LLVM::CallOp>(
                                 loc,
                                 TypeRange{regionUntypedPtrType},
                                 localRewriter.getStringAttr(
                                     "C_INTERFACE_GetMainRegion"),
                                 ValueRange{application.getResult()})
                             .getResult();

            return WalkResult::interrupt();
        }
    });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    patterns.add<
        ChannelOpLowering,
        InstantiateOpLowering_BUT_WE_NEED_TO_CALL_IT_SOMETHING_ELSE_FOR_SOME_REASON>(
        highLevelConverter,
        patterns.getContext());

    patterns.add<
        PushOpLowering,
        PullOpLowering,
        OperatorOpLowering_BUT_WE_NEED_TO_CALL_IT_SOMETHING_ELSE_FOR_SOME_REASON>(
        runtimeConverter,
        patterns.getContext());

    target.addLegalDialect<
        LLVM::LLVMDialect,
        cf::ControlFlowDialect,
        BuiltinDialect,
        func::FuncDialect>();
    target.addIllegalDialect<DfgDialect>();

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertDfgToLLVMPass()
{
    return std::make_unique<ConvertDfgToLLVMPass>();
}
