/// Implementation of DfgToAsync pass.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#include "dfg-mlir/Conversion/DfgToFunc/DfgToFunc.h"

#include "../PassDetails.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
        // TODO(feliix42): Is this check even necessary?
        if (!op.isExternal()) {
            rewriter.inlineRegionBefore(
                op.getBody(),
                genFuncOp.getBody(),
                genFuncOp.end());

            // add return op at the bottom of the body
            rewriter.setInsertionPointToEnd(&genFuncOp.getBody().back());
            auto returnOp = rewriter.create<func::ReturnOp>(
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
        // TODO(feiix42): Match for offloaded instantiate and DON'T lower
        rewriter.replaceOpWithNewOp<func::CallOp>(
            op,
            op.getCallee(),
            ArrayRef<Type>(),
            op.getOperands());

        // rewriter.eraseOp(op);

        return success();
    }
};

void ConvertDfgToFuncPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    target.addLegalDialect<BuiltinDialect, func::FuncDialect>();
    target.addLegalDialect<dfg::DfgDialect>();
    target.addIllegalOp<dfg::InstantiateOp, dfg::OperatorOp>();

    patterns.add<OperatorOpLowering, InstantiateOpLowering>(&getContext());

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