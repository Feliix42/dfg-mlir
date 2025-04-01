/// Implementation of ArithIndexToVitis pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/ArithIndexToVitis/ArithIndexToVitis.h"

#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "dfg-mlir/Dialect/vitis/IR/Dialect.h"
#include "dfg-mlir/Dialect/vitis/IR/Ops.h"
#include "dfg-mlir/Dialect/vitis/IR/Types.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
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
#define GEN_PASS_DEF_CONVERTARITHINDEXTOVITIS
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;

namespace {
struct ConvertArithConstant : OpConversionPattern<arith::ConstantOp> {

    using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

    ConvertArithConstant(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<arith::ConstantOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        arith::ConstantOp op,
        arith::ConstantOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto constantAttr = op.getValue();

        auto variableOp = rewriter.replaceOpWithNewOp<vitis::VariableOp>(
            op,
            op.getType(),
            /*init*/ constantAttr,
            /*is_const*/ true);
        op.getResult().replaceAllUsesWith(variableOp.getResult());

        return success();
    }
};

template<typename OpFrom, typename OpTo>
struct ConvertArithBinaryOp : OpConversionPattern<OpFrom> {

    using OpConversionPattern<OpFrom>::OpConversionPattern;

    ConvertArithBinaryOp(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<OpFrom>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        OpFrom op,
        OpFrom::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<OpTo>(op, op.getLhs(), op.getRhs());
        return success();
    }
};

template<typename OpFrom, typename OpTo>
struct ConvertArithCastOp : OpConversionPattern<OpFrom> {

    using OpConversionPattern<OpFrom>::OpConversionPattern;

    ConvertArithCastOp(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<OpFrom>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        OpFrom op,
        OpFrom::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<OpTo>(
            op,
            op.getOut().getType(),
            op.getIn());
        return success();
    }
};
} // namespace

void mlir::populateArithIndexToVitisConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<ConvertArithConstant>(typeConverter, patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::AddIOp, vitis::ArithAddOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::AddFOp, vitis::ArithAddOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<index::AddOp, vitis::ArithAddOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::SubIOp, vitis::ArithSubOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::SubFOp, vitis::ArithSubOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<index::SubOp, vitis::ArithSubOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::MulIOp, vitis::ArithMulOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::MulFOp, vitis::ArithMulOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<index::MulOp, vitis::ArithMulOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::DivFOp, vitis::ArithDivOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::RemUIOp, vitis::ArithRemOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::RemSIOp, vitis::ArithRemOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::RemFOp, vitis::ArithRemOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<index::RemUOp, vitis::ArithRemOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<index::RemSOp, vitis::ArithRemOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::AndIOp, vitis::ArithAndOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::OrIOp, vitis::ArithOrOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithCastOp<arith::IndexCastUIOp, vitis::ArithCastOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithCastOp<arith::UIToFPOp, vitis::ArithCastOp>>(
        typeConverter,
        patterns.getContext());
}

namespace {
struct ConvertArithIndexToVitisPass : public impl::ConvertArithIndexToVitisBase<
                                          ConvertArithIndexToVitisPass> {
    void runOnOperation() final;
};
} // namespace

void ConvertArithIndexToVitisPass::runOnOperation()
{
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateArithIndexToVitisConversionPatterns(converter, patterns);

    target.addLegalDialect<vitis::VitisDialect>();
    target.addIllegalDialect<arith::ArithDialect, index::IndexDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertArithIndexToVitisPass()
{
    return std::make_unique<ConvertArithIndexToVitisPass>();
}
