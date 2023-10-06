/// Implementation of the DfgToFunc pass.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Conversion/DfgToFunc/DfgToFunc.h"

#include "../PassDetails.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::dfg;

namespace {
struct ConvertDfgToFuncPass
        : public ConvertDfgToFuncBase<ConvertDfgToFuncPass> {
    void runOnOperation() final;
};
} // namespace

struct OperatorOpLowering : public OpConversionPattern<OperatorOp> {
    using OpConversionPattern<OperatorOp>::OpConversionPattern;

    OperatorOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<OperatorOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        OperatorOp op,
        OperatorOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext* context = rewriter.getContext();
        StringRef operatorName = op.getSymName();
        FunctionType funcTy = op.getFunctionType();
        auto old_inputs = funcTy.getInputs();
        auto old_outputs = funcTy.getResults();

        std::vector<Type> inputs = old_inputs.vec();
        inputs.insert(inputs.end(), old_outputs.begin(), old_outputs.end());
        FunctionType newFuncTy = FunctionType::get(context, inputs, {});

        auto genFuncOp =
            rewriter.create<func::FuncOp>(loc, operatorName, newFuncTy);
        // NOTE(feliix42): Is this check even necessary?
        if (!op.isExternal()) {
            rewriter.inlineRegionBefore(
                op.getBody(),
                genFuncOp.getBody(),
                genFuncOp.end());

            // add return op at the bottom of the body
            rewriter.setInsertionPointToEnd(&genFuncOp.getBody().back());
            rewriter.create<func::ReturnOp>(
                genFuncOp.getBody().back().back().getLoc(),
                ValueRange());
        }

        rewriter.eraseOp(op);

        return success();
    }
};

struct InstantiateOpLowering : public OpConversionPattern<InstantiateOp> {
    using OpConversionPattern<InstantiateOp>::OpConversionPattern;

    InstantiateOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<InstantiateOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        InstantiateOp op,
        InstantiateOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // don't lower offloaded functions
        if (op.getOffloaded()) return failure();

        rewriter.replaceOpWithNewOp<func::CallOp>(
            op,
            op.getCallee(),
            ArrayRef<Type>(),
            op.getOperands());

        return success();
    }
};

struct PushOpLowering : public OpConversionPattern<PushOp> {
    using OpConversionPattern<PushOp>::OpConversionPattern;

    PushOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<PushOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        PushOp op,
        PushOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // create function name for PushOp in LLVM IR
        std::string funcName = "push_";
        llvm::raw_string_ostream funcNameStream(funcName);
        // FIXME(feliix42): Is the op index correct?
        op.getOperand(0).getType().print(funcNameStream);

        SymbolRefAttr pushFuncName =
            SymbolRefAttr::get(op.getContext(), funcNameStream.str());
        Type boolReturnVal = rewriter.getI1Type();

        // FIXME(feliix42): Change this to llvm.call!!
        rewriter.create<func::CallOp>(
            op.getLoc(),
            pushFuncName,
            ArrayRef<Type>(boolReturnVal),
            op.getOperands());
        rewriter.eraseOp(op);

        return success();
    }
};

struct PullOpLowering : public OpConversionPattern<PullOp> {
    using OpConversionPattern<PullOp>::OpConversionPattern;

    PullOpLowering(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<PullOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        PullOp op,
        PullOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // create function name for PullOp in LLVM IR
        std::string funcName = "pull_";
        llvm::raw_string_ostream funcNameStream(funcName);
        op.getOperand().getType().print(funcNameStream);

        SymbolRefAttr pullFuncName =
            SymbolRefAttr::get(op.getContext(), funcNameStream.str());
        Type boolReturnVal = rewriter.getI1Type();
        std::vector<Type> structTypes{op.getType(), boolReturnVal};

        // create the struct type that models the result
        Type returnedStruct =
            LLVM::LLVMStructType::getLiteral(op.getContext(), structTypes);

        // rewrite the pull operation as llvm.call
        rewriter.create<func::CallOp>(
            op.getLoc(),
            pullFuncName,
            ArrayRef<Type>(returnedStruct),
            op.getOperand());
        rewriter.eraseOp(op);

        return success();
    }
};

void ConvertDfgToFuncPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    target.addLegalDialect<BuiltinDialect, func::FuncDialect>();
    target.addLegalDialect<DfgDialect>();
    target.addIllegalOp<OperatorOp, PushOp>();
    target.addDynamicallyLegalOp<InstantiateOp>(
        [](InstantiateOp op) { return op.getOffloaded(); });

    patterns.add<OperatorOpLowering, InstantiateOpLowering, PushOpLowering>(
        &getContext());

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertDfgToFuncPass()
{
    return std::make_unique<ConvertDfgToFuncPass>();
}