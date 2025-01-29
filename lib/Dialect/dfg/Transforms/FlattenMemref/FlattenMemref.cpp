/// Implements the dfg dialect ops bufferization.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Dialect/dfg/Transforms/FlattenMemref/FlattenMemref.h"

#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

namespace mlir {
namespace dfg {
#define GEN_PASS_DEF_DFGFLATTENMEMREF
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"
} // namespace dfg
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;

namespace {} // namespace

void mlir::dfg::populateFlattenMemrefPatterns(RewritePatternSet &patterns) {}

namespace {
struct DfgFlattenMemrefPass
        : public dfg::impl::DfgFlattenMemrefBase<DfgFlattenMemrefPass> {
    void runOnOperation() override;
};
} // namespace

void DfgFlattenMemrefPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateFlattenMemrefPatterns(patterns);

    target.addLegalDialect<DfgDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::dfg::createDfgFlattenMemrefPass()
{
    return std::make_unique<DfgFlattenMemrefPass>();
}
