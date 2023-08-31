/// Implementation of DfgToAsync pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@mailbox.tu-dresden.de)

#include "dfg-mlir/Conversion/DfgToAsync/DfgToAsync.h"

#include "../PassDetails.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::dfg;

namespace {}

void mlir::populateDfgToAsyncConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{}

namespace {
struct ConvertDfgToAsyncPass
        : public ConvertDfgToAsyncBase<ConvertDfgToAsyncPass> {
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
