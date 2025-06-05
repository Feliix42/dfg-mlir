/// Implementation of ArithIndexToEmitHLS pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/ArithIndexToEmitHLS/ArithIndexToEmitHLS.h"

#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "dfg-mlir/Dialect/emitHLS/IR/Dialect.h"
#include "dfg-mlir/Dialect/emitHLS/IR/Ops.h"
#include "dfg-mlir/Dialect/emitHLS/IR/Types.h"
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
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/Process.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_CONVERTARITHINDEXTOEMITHLS
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

        auto variableOp = rewriter.replaceOpWithNewOp<emitHLS::VariableOp>(
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

struct ConvertArithCompare : OpConversionPattern<arith::CmpIOp> {
    using OpConversionPattern<arith::CmpIOp>::OpConversionPattern;

    ConvertArithCompare(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<arith::CmpIOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        arith::CmpIOp op,
        arith::CmpIOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        return success();
    }
};

struct ConvertArithSelect : OpConversionPattern<arith::SelectOp> {
    using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

    ConvertArithSelect(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<arith::SelectOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        arith::SelectOp op,
        arith::SelectOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
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

void mlir::populateArithIndexToEmitHLSConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<ConvertArithConstant>(typeConverter, patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::AddIOp, emitHLS::ArithAddOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::AddFOp, emitHLS::ArithAddOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<index::AddOp, emitHLS::ArithAddOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::SubIOp, emitHLS::ArithSubOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::SubFOp, emitHLS::ArithSubOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<index::SubOp, emitHLS::ArithSubOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::MulIOp, emitHLS::ArithMulOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::MulFOp, emitHLS::ArithMulOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<index::MulOp, emitHLS::ArithMulOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::DivFOp, emitHLS::ArithDivOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::RemUIOp, emitHLS::ArithRemOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::RemSIOp, emitHLS::ArithRemOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::RemFOp, emitHLS::ArithRemOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<index::RemUOp, emitHLS::ArithRemOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<index::RemSOp, emitHLS::ArithRemOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::AndIOp, emitHLS::ArithAndOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithBinaryOp<arith::OrIOp, emitHLS::ArithOrOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithCompare>(typeConverter, patterns.getContext());
    patterns.add<ConvertArithSelect>(typeConverter, patterns.getContext());
    patterns
        .add<ConvertArithCastOp<arith::IndexCastUIOp, emitHLS::ArithCastOp>>(
            typeConverter,
            patterns.getContext());
    patterns.add<ConvertArithCastOp<arith::UIToFPOp, emitHLS::ArithCastOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertArithCastOp<arith::ExtSIOp, emitHLS::ArithCastOp>>(
        typeConverter,
        patterns.getContext());
}

namespace {
struct ConvertArithIndexToEmitHLSPass
        : public impl::ConvertArithIndexToEmitHLSBase<
              ConvertArithIndexToEmitHLSPass> {
    void runOnOperation() final;
};
} // namespace

void ConvertArithIndexToEmitHLSPass::runOnOperation()
{
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateArithIndexToEmitHLSConversionPatterns(converter, patterns);

    target.addLegalDialect<emitHLS::emitHLSDialect>();
    target.addIllegalDialect<arith::ArithDialect, index::IndexDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertArithIndexToEmitHLSPass()
{
    return std::make_unique<ConvertArithIndexToEmitHLSPass>();
}
