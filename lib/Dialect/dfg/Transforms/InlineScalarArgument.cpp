/// Implementation of InlineScalarArgument transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Dialect/dfg/Transforms/InlineScalarArgument.h"

#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/IRMapping.h>

namespace mlir {
namespace linalg {
#define GEN_PASS_DEF_LINALGINLINESCALARARGUMENT
#include "dfg-mlir/Dialect/dfg/Transforms/Passes.h.inc"
} // namespace linalg
} // namespace mlir

using namespace mlir;
using namespace linalg;

namespace {
struct UpdateGenericOp : public OpRewritePattern<GenericOp> {
    UpdateGenericOp(MLIRContext* context)
            : OpRewritePattern<GenericOp>(context){};

    LogicalResult
    matchAndRewrite(GenericOp op, PatternRewriter &rewriter) const override
    {
        auto inputs = op.getInputs();
        auto outputs = op.getOutputs();
        auto affineMaps = op.getIndexingMapsArray();
        Block* block = op.getBlock();
        // Get new inputs and affine maps
        SmallVector<Value> newInputs;
        SmallVector<AffineMap> newMaps;
        SmallVector<Value> keepUseArgs;
        IRMapping mapper;
        for (auto [idx, zip] : llvm::enumerate(llvm::zip(inputs, affineMaps))) {
            auto [input, affineMap] = zip;
            if (!isa<ShapedType>(input.getType())) {
                mapper.map(block->getArgument(idx), input);
            } else {
                newInputs.push_back(input);
                newMaps.push_back(affineMap);
                keepUseArgs.push_back(block->getArgument(idx));
            }
        }
        // The outputs
        for (unsigned i = 0; i < outputs.size(); ++i) {
            auto idx = i + inputs.size();
            newMaps.push_back(affineMaps[idx]);
            keepUseArgs.push_back(block->getArgument(idx));
        }
        // Replace with new generic
        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op,
            newInputs,
            op.getOutputs(),
            newMaps,
            op.getIteratorTypesArray(),
            /*doc*/ rewriter.getStringAttr(""),
            /*libraryCall*/ rewriter.getStringAttr(""),
            [&](OpBuilder &opBuilder, Location loc, ValueRange blockArgs) {
                for (auto [oldArg, newArg] : llvm::zip(keepUseArgs, blockArgs))
                    mapper.map(oldArg, newArg);
                for (auto &opOrig : block->getOperations())
                    opBuilder.clone(opOrig, mapper);
            });
        return success();
    }
};
} // namespace

void mlir::linalg::populateInlineScalarArgumentConversionPatterns(
    RewritePatternSet &patterns)
{
    patterns.add<UpdateGenericOp>(patterns.getContext());
}

namespace {
struct LinalgInlineScalarArgumentPass
        : public linalg::impl::LinalgInlineScalarArgumentBase<
              LinalgInlineScalarArgumentPass> {
    void runOnOperation() override;
};
} // namespace

void LinalgInlineScalarArgumentPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateInlineScalarArgumentConversionPatterns(patterns);

    target.markUnknownOpDynamicallyLegal([](Operation* op) { return true; });
    target.addDynamicallyLegalOp<GenericOp>([](GenericOp op) {
        for (auto input : op.getInputs())
            if (!isa<ShapedType>(input.getType())) return false;
        return true;
    });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::linalg::createLinalgInlineScalarArgumentPass()
{
    return std::make_unique<LinalgInlineScalarArgumentPass>();
}
