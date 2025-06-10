/// Implementation of MathToEmitHLS pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/MathToEmitHLS/MathToEmitHLS.h"

#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "dfg-mlir/Dialect/emitHLS/IR/Dialect.h"
#include "dfg-mlir/Dialect/emitHLS/IR/Ops.h"
#include "dfg-mlir/Dialect/emitHLS/IR/Types.h"
#include "mlir/Dialect/Math/IR/Math.h"
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
#define GEN_PASS_DEF_CONVERTMATHTOEMITHLS
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;

namespace {
template<typename OpFrom, typename OpTo>
struct ConvertMathUnaryOp : OpConversionPattern<OpFrom> {

    using OpConversionPattern<OpFrom>::OpConversionPattern;

    ConvertMathUnaryOp(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<OpFrom>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        OpFrom op,
        typename OpFrom::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<OpTo>(op, op.getOperand());
        return success();
    }
};
} // namespace

void mlir::populateMathToEmitHLSConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<ConvertMathUnaryOp<math::SinOp, emitHLS::MathSinOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertMathUnaryOp<math::CosOp, emitHLS::MathCosOp>>(
        typeConverter,
        patterns.getContext());
}

namespace {
struct ConvertMathToEmitHLSPass
        : public impl::ConvertMathToEmitHLSBase<ConvertMathToEmitHLSPass> {
    void runOnOperation() final;
};
} // namespace

void ConvertMathToEmitHLSPass::runOnOperation()
{
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateMathToEmitHLSConversionPatterns(converter, patterns);

    target.addLegalDialect<emitHLS::emitHLSDialect>();
    target.addIllegalDialect<math::MathDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertMathToEmitHLSPass()
{
    return std::make_unique<ConvertMathToEmitHLSPass>();
}
