/// Implementation of ScfToEmitHLS pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "dfg-mlir/Conversion/ScfToEmitHLS/ScfToEmitHLS.h"

#include "dfg-mlir/Conversion/Utils.h"
#include "dfg-mlir/Dialect/dfg/IR/Dialect.h"
#include "dfg-mlir/Dialect/dfg/IR/Ops.h"
#include "dfg-mlir/Dialect/dfg/IR/Types.h"
#include "dfg-mlir/Dialect/emitHLS/IR/Dialect.h"
#include "dfg-mlir/Dialect/emitHLS/IR/Ops.h"
#include "dfg-mlir/Dialect/emitHLS/IR/Types.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/APInt.h"

#include <cstddef>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Process.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_CONVERTSCFTOEMITHLS
#include "dfg-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::emitHLS;

namespace {
struct ConvertForOp : OpConversionPattern<scf::ForOp> {
    using OpConversionPattern<scf::ForOp>::OpConversionPattern;

    ConvertForOp(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<scf::ForOp>(typeConverter, context){};

    LogicalResult matchAndRewrite(
        scf::ForOp op,
        scf::ForOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto lb = op.getLowerBound();
        auto ub = op.getUpperBound();
        auto step = op.getStep();

        bool hasLoopInside = false;
        op->walk([&](Operation* childOp) {
            if (childOp != op && isa<scf::ForOp>(childOp)) {
                hasLoopInside = true;
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });

        Attribute lbVal, ubVal, stepVal;
        if (auto defOp = dyn_cast<emitHLS::VariableOp>(lb.getDefiningOp()))
            lbVal = defOp.getInitAttr();
        if (auto defOp = dyn_cast<emitHLS::VariableOp>(ub.getDefiningOp()))
            ubVal = defOp.getInitAttr();
        if (auto defOp = dyn_cast<emitHLS::VariableOp>(step.getDefiningOp()))
            stepVal = defOp.getInitAttr();

        if (op.getResults().size() != 0)
            return rewriter.notifyMatchFailure(
                loc,
                "No support for scf.for op with results");

        auto emitHLSFor = rewriter.create<ForOp>(loc, lbVal, ubVal, stepVal);
        rewriter.setInsertionPointToEnd(&emitHLSFor.getBody().front());
        if (!hasLoopInside)
            rewriter.create<PragmaPipelineOp>(emitHLSFor.getLoc());
        IRMapping mapper;
        mapper.map(op.getInductionVar(), emitHLSFor.getInductionVar());

        // Copy scf.for contents into new emitHLS.for
        for (auto &opi : op.getRegion().getOps()) {
            if (isa<scf::YieldOp>(opi)) continue;
            auto clonedOp = rewriter.clone(opi, mapper);
            if (auto nestedFor = dyn_cast<scf::ForOp>(clonedOp)) {
                scf::ForOpAdaptor nestedAdaptor(nestedFor);
                if (failed(matchAndRewrite(nestedFor, nestedAdaptor, rewriter)))
                    return rewriter.notifyMatchFailure(
                        nestedFor.getLoc(),
                        "Failed to process nested for.");
            }
        }

        rewriter.replaceOp(op, emitHLSFor);
        return success();
    }
};
} // namespace

void mlir::populateScfToEmitHLSConversionPatterns(
    TypeConverter typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<ConvertForOp>(typeConverter, patterns.getContext());
}

namespace {
struct ConvertScfToEmitHLSPass
        : public impl::ConvertScfToEmitHLSBase<ConvertScfToEmitHLSPass> {
    void runOnOperation() final;
};
} // namespace

void ConvertScfToEmitHLSPass::runOnOperation()
{
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    populateScfToEmitHLSConversionPatterns(converter, patterns);

    target.addLegalDialect<emitHLS::emitHLSDialect>();
    target.addIllegalDialect<scf::SCFDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertScfToEmitHLSPass()
{
    return std::make_unique<ConvertScfToEmitHLSPass>();
}
