/// Implementation of InlineRegion transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/CommandLine.h"

#include <circt/Support/LLVM.h>

namespace mlir {
namespace dfg {
#define GEN_PASS_DEF_DFGINLINEREGION
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"
} // namespace dfg
} // namespace mlir

using namespace mlir;
using namespace dfg;

namespace {} // namespace

void mlir::dfg::populateInlineRegionConversionPatterns(
    RewritePatternSet &patterns)
{}

namespace {
enum class InlineStrategy { Smart, Full };
struct DfgInlineRegionPass
        : public dfg::impl::DfgInlineRegionBase<DfgInlineRegionPass> {
    void runOnOperation() override;
};
} // namespace

void DfgInlineRegionPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateInlineRegionConversionPatterns(patterns);

    auto moduleOp = getOperation();

    target.addLegalDialect<DfgDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation* op) {
        return op->getDialect()->getNamespace() != "dfg";
    });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::dfg::createDfgInlineRegionPass()
{
    return std::make_unique<DfgInlineRegionPass>();
}
