/// Implementation of DfgToAsync pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/DfgToAsync/DfgToAsync.h"

#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTDFGTOASYNC
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;

namespace {}

void mlir::populateDfgToAsyncConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{}

namespace {
struct ConvertDfgToAsyncPass
        : public impl::ConvertDfgToAsyncBase<ConvertDfgToAsyncPass> {
    void runOnOperation() final;
};
} // namespace

void ConvertDfgToAsyncPass::runOnOperation()
{
    TypeConverter converter;
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    target.addLegalDialect<BuiltinDialect, async::AsyncDialect>();
    target.addIllegalDialect<dfg::DfgDialect>();

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertDfgToAsyncPass()
{
    return std::make_unique<ConvertDfgToAsyncPass>();
}
